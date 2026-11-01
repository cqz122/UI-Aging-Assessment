# -*- coding: utf-8 -*-
"""召回率计算：拿系统结果和人工标注对比。

为什么要这个脚本：
    12 条条目里，有的靠程序测量（准），有的靠大模型判断（可能漏判）。
    只靠"感觉挺准"没法交代，所以必须有一份人工标注做基准，
    算出"该报的问题报出来了吗"——也就是召回率。

正例定义：人工标注为"不通过"的条目。
    召回率 = 命中的正例 / 全部正例        （漏报越少越高）
    精确率 = 命中的正例 / 系统判不通过数   （误报越少越高）
    覆盖率 = 系统给出明确结论的条目 / 总条目（"无法判断"不算结论）

用法：
    python eval/recall.py
    python eval/recall.py --labels eval/labels --results outputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
POSITIVE = "不通过"


def load_labels(label_dir: str) -> Dict[str, Dict[str, str]]:
    """读取全部标注文件：{图片名: {条目编号: 结论}}。"""
    labels: Dict[str, Dict[str, str]] = {}
    for file in sorted(Path(label_dir).glob("*.json")):
        with open(file, encoding="utf-8") as f:
            labels.update(json.load(f))
    return labels


def load_results(result_dir: str) -> Dict[str, Dict[str, str]]:
    """读取 outputs 下的结果 JSON：{图片名: {条目编号: 结论}}。"""
    results: Dict[str, Dict[str, str]] = {}
    for file in sorted(Path(result_dir).glob("*_result.json")):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        name = os.path.basename(data.get("image", file.name))
        results[name] = {item["id"]: item["status"] for item in data.get("items", [])}
    return results


def evaluate(
    labels: Dict[str, Dict[str, str]], results: Dict[str, Dict[str, str]]
) -> Dict[str, object]:
    """逐条比对，统计 TP / FP / FN。"""
    rows: List[Dict[str, str]] = []
    tp = fp = fn = judged = total = 0

    for image, label_items in labels.items():
        system_items = results.get(image, {})
        if not system_items:
            rows.append({"image": image, "id": "-", "label": "-", "system": "-", "note": "没有找到结果文件"})
            continue

        for item_id, expected in label_items.items():
            actual = system_items.get(item_id, "未检测")
            total += 1
            if actual in ("通过", "不通过"):
                judged += 1

            if expected == POSITIVE and actual == POSITIVE:
                tp += 1
                note = "命中"
            elif expected == POSITIVE:
                fn += 1
                note = "漏报"
            elif actual == POSITIVE:
                fp += 1
                note = "误报"
            elif actual == "无法判断":
                note = "未判定"
            else:
                note = "一致"

            rows.append(
                {"image": image, "id": item_id, "label": expected, "system": actual, "note": note}
            )

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    coverage = judged / total if total else 0.0

    return {
        "rows": rows,
        "tp": tp, "fp": fp, "fn": fn,
        "total": total,
        "recall": recall,
        "precision": precision,
        "coverage": coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="计算检测召回率")
    parser.add_argument("--labels", default=str(BASE_DIR / "eval" / "labels"))
    parser.add_argument("--results", default=str(BASE_DIR / "outputs"))
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if not labels:
        print(f"没有找到标注文件：{args.labels}")
        return 1

    results = load_results(args.results)
    if not results:
        print(f"没有找到结果文件：{args.results}（先运行 main.py 生成结果）")
        return 1

    report = evaluate(labels, results)

    print(f"{'图片':<18}{'条目':<6}{'人工标注':<10}{'系统结论':<10}说明")
    print("-" * 60)
    for row in report["rows"]:
        print(
            f"{row['image'][:16]:<18}{row['id']:<6}{row['label']:<10}"
            f"{row['system']:<10}{row['note']}"
        )

    print("-" * 60)
    print(f"正例（人工判不通过）{report['tp'] + report['fn']} 条，"
          f"命中 {report['tp']} 条，漏报 {report['fn']} 条，误报 {report['fp']} 条")
    print(f"召回率：{report['recall']:.1%}")
    print(f"精确率：{report['precision']:.1%}")
    print(f"判定覆盖率：{report['coverage']:.1%}（“无法判断”不计入结论）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
