# -*- coding: utf-8 -*-
"""调度引擎：读条目 → 跑工具或问模型 → 汇总结果。

这是"组长维护"的部分，组员一般不需要改这里。
新增检测条目的同学只需要改 prompts/items.py；
需要新测量能力的同学加一个 tools/xxx.py。

条目支持三种判定方式：
    method="tool"   完全由程序测量，按 judge 里的阈值比对，不花 API 额度
    method="llm"    把事实卡和截图交给大模型，由模型判断
    method="hybrid" 先跑工具，把测量结果一起给模型，由模型结合语境下结论
"""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs import config
from core.context import PageContext
from core.llm import LLMClient, LLMError

# 系统提示词：所有条目共用，负责"把模型的角色和纪律定死"
SYSTEM_PROMPT = """\
你是移动应用适老化检测专家，负责按照国家标准与企业规范逐条审核 App 界面。

每次判定你会收到两样东西：
1. 一张完整的界面截图（可以看到颜色、图标、按钮形状、弹窗、遮挡关系）；
2. 一段由 OCR 与像素测量生成的"页面事实卡"（全部可见文字及其坐标，以及字号、行距等数值）。

两者冲突时，以事实卡里的数值为准——那是由程序测量得到的精确值；
截图用来判断视觉与语义问题（例如"这个图标没有文字标签""弹窗挡住了主按钮"）。

判定纪律：
1. 结论只能是三选一：通过 / 不通过 / 无法判断。
2. 判"不通过"必须给出具体依据：引用截图里的位置（如"页面底部""右上角弹窗"）
   或事实卡里的原文，不能只说"存在风险"。
3. 截图和事实卡都看不出来的信息，选"无法判断"，并说明还缺什么，不要猜测。
4. 建议要具体可执行，给出改法（例如"把按钮文字从 12dp 提到 18dp"），
   不要写"优化体验""提升易用性"这类空话。
5. 只输出 JSON，不要输出解释性文字，也不要用 Markdown 代码块包裹。

输出格式（严格按这个来，字段名不要改）：
{"item_id": "A01",
 "status": "通过|不通过|无法判断",
 "evidence": ["证据1", "证据2"],
 "reason": "一句话说明判定理由",
 "suggestion": "具体改法，判通过时可为空字符串"}
"""

# suggestion 写 "auto" 时，用这个提示词让模型现写一句改法
SUGGEST_SYSTEM = """\
你是移动应用适老化改造顾问，服务对象是没有设计背景的学生团队。
根据给出的条目、判定和证据，写一句可以直接执行的修改建议。
要求：不超过 40 字，给出具体改法（例如"把浅灰提示文字改为 #666"），不要解释原因，不要加引号。"""


class Agent:
    """检测调度器。"""

    def __init__(self, llm: Optional[LLMClient] = None, mock: bool = False, dpi: Optional[float] = None):
        self.llm = llm or LLMClient()
        self.mock = mock  # True 时不调用模型，用于离线调试
        self.dpi = dpi

    # ------------------------------------------------------------------ #
    def run(self, image_path: str, only: Optional[List[str]] = None) -> Dict[str, Any]:
        """对一张截图跑完全部条目。"""
        page = PageContext.load(image_path, dpi=self.dpi)
        items = load_items()

        if only:
            wanted = {code.strip().upper() for code in only}
            items = [item for item in items if item["id"].upper() in wanted]

        results: List[Dict[str, Any]] = []
        for item in items:
            results.append(self._judge(item, page))

        summary = {"通过": 0, "不通过": 0, "无法判断": 0}
        for result in results:
            summary[result["status"]] = summary.get(result["status"], 0) + 1

        return {
            "image": image_path,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size": [page.width, page.height],
            "dpi": page.dpi,
            "line_count": len(page.lines),
            "items": results,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    def _judge(self, item: Dict[str, Any], page: PageContext) -> Dict[str, Any]:
        """按 method 分派，并统一包装成同一种结果结构。"""
        method = item.get("method", "llm")
        try:
            if method == "tool":
                outcome = self._judge_tool(item, page)
            elif method == "hybrid":
                outcome = self._judge_llm(item, page, extra=self._tool_evidence(item, page))
            else:
                outcome = self._judge_llm(item, page)
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响其他条目
            outcome = {
                "status": "无法判断",
                "evidence": [],
                "reason": f"本条执行失败：{type(exc).__name__}: {exc}",
                "suggestion": "",
            }

        outcome.update(
            {
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "condition": item.get("condition", ""),
                "method": method,
            }
        )
        # suggestion 写 "auto" 时，判不通过就让模型现写一条建议
        if outcome.get("status") == "不通过" and item.get("suggestion") == "auto":
            outcome["suggestion"] = self._suggest(item, outcome)
        outcome["item_id"] = outcome["id"]
        return outcome

    # ------------------------------------------------------------------ #
    def _suggest(self, item: Dict[str, Any], outcome: Dict[str, Any]) -> str:
        """让模型根据判定结果写一句修改建议（suggestion="auto" 时使用）。"""
        if self.mock or not self.llm.available:
            return ""
        user = (
            f"条目：{item.get('title')}\n"
            f"判定：{outcome.get('reason')}\n"
            f"证据：{'；'.join(outcome.get('evidence') or [])}\n"
            "请给一句具体可执行的修改建议，不超过 40 字。"
        )
        try:
            return clean_suggestion(self.llm.chat(SUGGEST_SYSTEM, user))
        except Exception:  # noqa: BLE001 - 建议是加分项，失败不影响结论
            return ""

    # ------------------------------------------------------------------ #
    def _judge_tool(self, item: Dict[str, Any], page: PageContext) -> Dict[str, Any]:
        """纯程序判定：跑工具 → 逐条取字段比阈值。

        judge 可以写一个条件，也可以写多个（默认全部满足才通过）：
            "judge": {"field": "low_count", "op": "==", "value": 0, ...}
            "judge": [ {...}, {...} ]          多个条件，全部满足才通过
            "judge": {"any": [ {...}, {...} ]} 多个条件，满足任意一个即通过
        """
        metrics = page.tool(item["tool"])
        rules, mode = parse_judge(item.get("judge"))

        evidence: List[str] = []
        passed_flags: List[bool] = []
        unknown: List[str] = []

        for rule in rules:
            field = rule.get("field", "")
            value = metrics.get(field)
            label = rule.get("label", field)
            unit = rule.get("unit", "")
            op = rule.get("op", ">=")
            target = rule.get("value", 0)

            if value is None or (isinstance(value, (int, float)) and value <= 0):
                unknown.append(label)
                continue

            ok = compare(value, op, target)
            passed_flags.append(ok)
            evidence.append(f"{label} {value}{unit}（阈值 {op} {target}{unit}）")

        if not passed_flags:
            return {
                "status": "无法判断",
                "evidence": [],
                "reason": f"工具 {item['tool']} 未能测出所需数值（页面上可能没有可测量的文本）",
                "suggestion": "",
            }

        if mode == "any":
            passed = any(passed_flags)
            reason = "至少一个条件满足" if passed else "全部条件都未达到"
        else:
            passed = all(passed_flags)
            reason = "条件全部满足" if passed else "有条件未达到"

        # 顺手把统计行数补进证据，报告更好读
        line_count = metrics.get("line_count")
        if isinstance(line_count, int) and line_count:
            evidence.append(f"共统计 {line_count} 行文字")

        return {
            "status": "通过" if passed else "不通过",
            "evidence": evidence,
            "reason": reason,
            "suggestion": "" if passed else item.get("suggestion", ""),
        }

    # ------------------------------------------------------------------ #
    def _judge_llm(self, item: Dict[str, Any], page: PageContext, extra: str = "") -> Dict[str, Any]:
        """交给大模型判断。"""
        if self.mock or not self.llm.available:
            return {
                "status": "无法判断",
                "evidence": [],
                "reason": "未配置大模型 Key（或使用了 --mock），本条需要模型判断，已跳过",
                "suggestion": "",
            }

        user = build_user_prompt(item, page, extra)
        payload = self.llm.chat_json(SYSTEM_PROMPT, user, image_path=page.image_path)
        return normalize(payload, item.get("id", ""))

    def _tool_evidence(self, item: Dict[str, Any], page: PageContext) -> str:
        """hybrid 条目：先跑工具，把结果作为补充证据交给模型。"""
        tool_name = item.get("tool")
        if not tool_name:
            return ""
        metrics = page.tool(tool_name)
        lines = [f"- {key}: {value}" for key, value in metrics.items() if key != "threshold_ratio"]
        return "【程序测量】\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# 提示词拼装
# --------------------------------------------------------------------------- #
def build_user_prompt(item: Dict[str, Any], page: PageContext, extra: str = "") -> str:
    """给模型的用户消息：条目要求 + 判定示例 + 程序测量 + 页面事实卡。"""
    parts = [
        f"【检测条目】{item.get('id')} {item.get('title')}",
        f"【适用条件】{item.get('condition', '无')}",
        f"【依据】{item.get('source', '无')}",
        f"【判定阈值】{item.get('threshold', '无（定性）')}",
        "",
        "【判定要求】",
        item.get("prompt", "").strip(),
        "",
    ]
    if item.get("examples"):
        parts.extend([format_examples(item), ""])
    if extra:
        parts.extend([extra, ""])
    parts.append(page.fact_card())
    return "\n".join(parts)


def format_examples(item: Dict[str, Any]) -> str:
    """把条目里的 examples 渲染成"输入 → 输出"的样例，教模型照格式回答。

    样例统一用 JSON 字符串，和最终输出格式一致，
    比用自然语言描述"你应该返回什么"更有效。
    """
    lines = ["【判定示例】照着示例的格式和尺度判断，不要照抄内容。"]
    for index, example in enumerate(item.get("examples", []), 1):
        lines.append("")
        lines.append(f"示例 {index}：")
        if example.get("input"):
            lines.append(f"输入：{str(example['input']).strip()}")
        output = example.get("output", {})
        payload = output if isinstance(output, dict) else {"status": str(output)}
        payload.setdefault("item_id", item.get("id", ""))
        lines.append(f"输出：{json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def normalize(payload: Dict[str, Any], item_id: str = "") -> Dict[str, Any]:
    """把模型输出规整成统一结构，防止字段缺失导致报告出错。"""
    status = str(payload.get("status") or payload.get("verdict") or "").strip()
    if status not in {"通过", "不通过", "无法判断"}:
        status = "无法判断"

    evidence = payload.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(e).strip() for e in evidence if str(e).strip()]

    reason = str(payload.get("reason", "")).strip()
    wrong_id = str(payload.get("item_id", "")).strip()
    if wrong_id and item_id and wrong_id != item_id:
        # 模型有时会串条目，记一笔，避免报告张冠李戴
        reason = f"{reason}（注意：模型返回的 item_id 是 {wrong_id}）"

    return {
        "status": status,
        "evidence": evidence[:5],
        "reason": reason,
        "suggestion": str(payload.get("suggestion", "")).strip(),
    }


def compare(value: float, op: str, target: float) -> bool:
    """支持 >= > <= < == 五种比较，方便条目里用声明式写法。"""
    value, target = float(value), float(target)
    return {
        ">=": value >= target,
        ">": value > target,
        "<=": value <= target,
        "<": value < target,
        "==": value == target,
    }.get(op, value >= target)


def parse_judge(judge: Any) -> tuple:
    """把 judge 统一成 (条件列表, 组合方式)。

        {"field": ...}               -> 单条件
        [{...}, {...}]               -> 多条件，全部满足才通过（all）
        {"any": [{...}, {...}]}      -> 多条件，满足任意一个即通过（any）
        {"all": [{...}, {...}]}      -> 同列表写法
    """
    if isinstance(judge, dict):
        if isinstance(judge.get("any"), list):
            return [r for r in judge["any"] if isinstance(r, dict)], "any"
        if isinstance(judge.get("all"), list):
            return [r for r in judge["all"] if isinstance(r, dict)], "all"
        return [judge], "all"
    if isinstance(judge, list):
        return [r for r in judge if isinstance(r, dict)], "all"
    return [], "all"


def clean_suggestion(text: str) -> str:
    """把模型返回的建议清理成一句话。

    模型偶尔会不听话，返回一整个 JSON，这里兼容一下：能解析出 suggestion 就用它，
    再剥掉多余的引号和换行，最后限长，避免报告被一段长文撑爆。
    """
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
            text = str(payload.get("suggestion") or payload.get("reason") or "").strip()
        except Exception:  # noqa: BLE001
            pass
    text = text.strip().strip('"').strip("“”").replace("\n", " ").strip()
    return text[:120]


# --------------------------------------------------------------------------- #
# 条目加载
# --------------------------------------------------------------------------- #
def load_items() -> List[Dict[str, Any]]:
    """扫描 prompts/ 目录，把每个条目文件里的 ITEM 收集起来。

    一人一条文件的好处：从 GitHub 网页上改也不会和别人冲突，
    有新条目只需要新建一个文件，不用动别人的文件。
    """
    folder = Path(config.PROMPT_DIR)
    items: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}

    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):  # _模板.py 之类不加载
            continue
        module = _import_file(path)
        item = getattr(module, "ITEM", None)
        if not isinstance(item, dict):
            print(f"[跳过] {path.name}：文件里没有 ITEM 字典")
            continue
        if not _validate(item, path):
            continue
        if item["id"] in seen:
            print(f"[跳过] {path.name}：编号 {item['id']} 已在 {seen[item['id']]} 中使用")
            continue
        seen[item["id"]] = path.name
        items.append(item)

    return items


def _import_file(path: Path):
    """按文件路径加载模块（文件名可以是中文）。"""
    module_name = "prompt_item_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(item: Dict[str, Any], path: Path) -> bool:
    """检查必填字段，字段不全时给出可读提示，而不是跑到一半报错。"""
    for key in ("id", "title", "method"):
        if not item.get(key):
            print(f"[跳过] {path.name}：缺少必填字段 {key}")
            return False

    method = item["method"]
    if method not in {"tool", "llm", "hybrid"}:
        print(f"[跳过] {path.name}：method 只能是 tool / llm / hybrid")
        return False

    if method in {"tool", "hybrid"}:
        from tools import TOOLS

        tool_name = item.get("tool")
        if tool_name not in TOOLS:
            print(f"[跳过] {path.name}：工具 “{tool_name}” 不存在，请检查 tools/ 下的文件名与 NAME 是否写对")
            return False

    if method in {"llm", "hybrid"} and not item.get("prompt", "").strip():
        print(f"[跳过] {path.name}：method 为 {method} 时必须写 prompt")
        return False

    if method == "tool" and not item.get("judge"):
        print(f"[跳过] {path.name}：method 为 tool 时必须写 judge（拿哪个字段、怎么比）")
        return False

    return True
