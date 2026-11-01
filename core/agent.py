# -*- coding: utf-8 -*-
"""调度引擎：读条目 → 跑工具或问模型 → 汇总结果。

这是"组长维护"的部分，组员一般不需要改这里。
新增检测条目的同学只需要改 prompts/items.py；
需要新测量能力的同学加一个 tools/xxx.py。

条目支持三种判定方式：
    method="tool"   完全由程序测量，按 judge 里的阈值比对，不花 API 额度
    method="llm"    把事实卡交给 DeepSeek，由模型判断
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
from core.llm import DeepSeekClient, DeepSeekError

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


class Agent:
    """检测调度器。"""

    def __init__(self, llm: Optional[DeepSeekClient] = None, mock: bool = False, dpi: Optional[float] = None):
        self.llm = llm or DeepSeekClient()
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
        outcome["item_id"] = outcome["id"]
        return outcome

    # ------------------------------------------------------------------ #
    def _judge_tool(self, item: Dict[str, Any], page: PageContext) -> Dict[str, Any]:
        """纯程序判定：跑工具 → 取字段 → 比阈值。"""
        rule = item.get("judge") or {}
        metrics = page.tool(item["tool"])
        field = rule.get("field", "")
        value = metrics.get(field)

        if value is None or (isinstance(value, (int, float)) and value <= 0):
            return {
                "status": "无法判断",
                "evidence": [],
                "reason": f"工具 {item['tool']} 未能测出 {field}（页面上可能没有可测量的文本）",
                "suggestion": "",
            }

        passed = compare(value, rule.get("op", ">="), rule.get("value", 0))
        label = rule.get("label", field)
        unit = rule.get("unit", "")

        evidence = [f"{label} {value}{unit}（阈值 {rule.get('op', '>=')} {rule.get('value')}{unit}）"]
        extra = metrics.get("line_count")
        if isinstance(extra, int) and extra:
            evidence.append(f"共统计 {extra} 行文字")

        return {
            "status": "通过" if passed else "不通过",
            "evidence": evidence,
            "reason": f"{label} {value}{unit}，{'达到' if passed else '未达到'} {rule.get('value')}{unit}",
            "suggestion": "" if passed else item.get("suggestion", ""),
        }

    # ------------------------------------------------------------------ #
    def _judge_llm(self, item: Dict[str, Any], page: PageContext, extra: str = "") -> Dict[str, Any]:
        """交给 DeepSeek 判断。"""
        if self.mock or not self.llm.available:
            return {
                "status": "无法判断",
                "evidence": [],
                "reason": "未配置 DEEPSEEK_API_KEY（或使用了 --mock），本条需要模型判断，已跳过",
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
            print(f"[跳过] {path.name}：工具 “{tool_name}” 不存在，请先在 tools/__init__.py 登记")
            return False

    if method in {"llm", "hybrid"} and not item.get("prompt", "").strip():
        print(f"[跳过] {path.name}：method 为 {method} 时必须写 prompt")
        return False

    if method == "tool" and not item.get("judge"):
        print(f"[跳过] {path.name}：method 为 tool 时必须写 judge（拿哪个字段、怎么比）")
        return False

    return True
