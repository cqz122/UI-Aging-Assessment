# -*- coding: utf-8 -*-
"""反馈文档生成：把检测结果写成给人看的 Markdown，并落一份 JSON。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from configs import config

VERDICT_ICON = {"通过": "✅ 通过", "不通过": "❌ 不通过", "无法判断": "⚠️ 无法判断"}
VERDICT_ORDER = {"不通过": 0, "无法判断": 1, "通过": 2}


def render_markdown(result: Dict[str, Any], facts: str = "") -> str:
    """生成 Markdown 反馈文档。"""
    items = sorted(result["items"], key=lambda i: VERDICT_ORDER.get(i["status"], 3))
    summary = result.get("summary", {})

    lines: List[str] = []
    lines.append(f"# {config.REPORT_TITLE}")
    lines.append("")
    lines.append(f"- **页面**：`{os.path.basename(result['image'])}`")
    lines.append(
        f"- **尺寸**：{result['size'][0]} × {result['size'][1]} px，dpi {result['dpi']:g}"
    )
    lines.append(f"- **检测时间**：{result['time']}")
    lines.append(f"- **识别文本行**：{result.get('line_count', 0)} 行")
    lines.append(
        f"- **结论**：不通过 {summary.get('不通过', 0)} 条，"
        f"无法判断 {summary.get('无法判断', 0)} 条，"
        f"通过 {summary.get('通过', 0)} 条"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for item in items:
        lines.append(f"### {item['id']} · {item['title']}")
        lines.append("")
        lines.append(f"**{VERDICT_ICON.get(item['status'], item['status'])}**")
        lines.append("")
        lines.append(f"- 依据：{item.get('source') or '—'}")
        if item.get("condition"):
            lines.append(f"- 适用条件：{item['condition']}")
        lines.append(f"- 判定：{item.get('reason') or '—'}")
        if item.get("evidence"):
            lines.append("- 证据：")
            for evidence in item["evidence"]:
                lines.append(f"  - {evidence}")
        if item.get("suggestion"):
            lines.append(f"- 建议：{item['suggestion']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 汇总建议")
    lines.append("")
    fixes = [item for item in items if item["status"] == "不通过" and item.get("suggestion")]
    if fixes:
        for index, item in enumerate(fixes, 1):
            lines.append(f"{index}. **{item['id']} {item['title']}**：{item['suggestion']}")
    else:
        lines.append("本次检测未发现需要修改的条目。")
    lines.append("")

    if facts:
        lines.append("---")
        lines.append("")
        lines.append("## 附：本次使用的页面事实卡")
        lines.append("")
        lines.append("```text")
        lines.append(facts)
        lines.append("```")
        lines.append("")

    lines.append(
        "> 本报告由程序测量与大模型判定自动生成，结论仅供设计参考，"
        "正式验收仍需人工复核。"
    )
    return "\n".join(lines)


def save(result: Dict[str, Any], facts: str = "", outdir: str | None = None) -> tuple:
    """保存 Markdown 与 JSON，返回两个路径。"""
    out_dir = outdir or str(config.OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(result["image"]))[0]

    md_path = os.path.join(out_dir, f"{stem}_反馈.md")
    json_path = os.path.join(out_dir, f"{stem}_result.json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result, facts))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return md_path, json_path


# --------------------------------------------------------------------------- #
# 条目清单导出
# --------------------------------------------------------------------------- #
def render_items_markdown(items: List[Dict[str, Any]]) -> str:
    """把散落在 prompts/ 里的条目汇总成一份清单，便于提交和评审。"""
    counts = {"tool": 0, "llm": 0, "hybrid": 0}
    for item in items:
        counts[item.get("method", "llm")] = counts.get(item.get("method", "llm"), 0) + 1

    lines: List[str] = ["# 检测条目清单", ""]
    lines.append(
        f"共 {len(items)} 条：程序测量 {counts.get('tool', 0)} 条，"
        f"模型判断 {counts.get('llm', 0)} 条，混合 {counts.get('hybrid', 0)} 条。"
    )
    lines.append("")
    lines.append("| 编号 | 条目 | 判定方式 | 依据 |")
    lines.append("| --- | --- | --- | --- |")
    for item in items:
        lines.append(
            f"| {item['id']} | {item['title']} | {item.get('method', 'llm')} | "
            f"{item.get('source', '—')} |"
        )
    lines.append("")

    for item in items:
        lines.append(f"## {item['id']} {item['title']}")
        lines.append("")
        lines.append(f"- 判定方式：{item.get('method', 'llm')}")
        if item.get("condition"):
            lines.append(f"- 适用条件：{item['condition']}")
        if item.get("source"):
            lines.append(f"- 依据：{item['source']}")
        if item.get("tool"):
            lines.append(f"- 使用工具：{item['tool']}")
        if item.get("judge"):
            rule = item["judge"]
            lines.append(
                f"- 判定规则：{rule.get('label', rule.get('field'))} "
                f"{rule.get('op')} {rule.get('value')}{rule.get('unit', '')}"
            )
        if item.get("prompt"):
            lines.append("")
            lines.append("判定要求：")
            lines.append("")
            lines.append("```text")
            lines.append(item["prompt"].strip())
            lines.append("```")
        lines.append("")

    return "\n".join(lines)


def export_items(items: List[Dict[str, Any]], outdir: str | None = None) -> str:
    """导出条目清单到文件，返回路径。"""
    out_dir = outdir or str(config.OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "条目清单.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_items_markdown(items))
    return path
