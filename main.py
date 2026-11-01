# -*- coding: utf-8 -*-
"""运行入口。

用法：
    python main.py examples/screenshots/demo.png
    python main.py 截图.png --items A01,A02
    python main.py 截图.png --mock          # 不调用模型，只跑程序测量
    python main.py --list-items             # 看有哪些条目
    python main.py --list-tools             # 看有哪些测量工具
    python main.py 截图.png --print-facts   # 额外打印页面事实卡
"""

from __future__ import annotations

import argparse
import sys

from core.agent import Agent, load_items
from core.report import save
from tools import list_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="适老化检测：一张截图，逐条判定，输出反馈文档"
    )
    parser.add_argument("image", nargs="?", help="截图路径")
    parser.add_argument("--items", default="", help="只跑指定条目，逗号分隔，如 A01,A05")
    parser.add_argument("--dpi", type=float, default=None, help="截图设备 dpi，默认取 configs/config.py")
    parser.add_argument("--mock", action="store_true", help="不调用大模型，仅跑程序测量（离线调试）")
    parser.add_argument("--out", default="", help="输出目录，默认 outputs/")
    parser.add_argument("--print-facts", action="store_true", help="在终端打印页面事实卡")
    parser.add_argument("--list-items", action="store_true", help="列出全部检测条目")
    parser.add_argument("--list-tools", action="store_true", help="列出全部测量工具")
    parser.add_argument("--export-items", action="store_true", help="把所有条目汇总成一份清单")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_items:
        print(f"{'编号':<6}{'方式':<10}{'条目'}")
        for item in load_items():
            print(f"{item['id']:<6}{item.get('method', 'llm'):<10}{item['title']}")
        return 0

    if args.list_tools:
        print(f"{'工具名':<16}{'说明'}")
        for tool in list_tools():
            print(f"{tool['name']:<16}{tool['description']}")
        return 0

    if args.export_items:
        from core.report import export_items

        items = load_items()
        path = export_items(items, args.out or None)
        print(f"已导出 {len(items)} 条到：{path}")
        return 0

    if not args.image:
        build_parser().print_help()
        return 1

    only = [code for code in args.items.split(",") if code.strip()] or None
    agent = Agent(mock=args.mock, dpi=args.dpi)
    result = agent.run(args.image, only=only)

    print(f"\n==> {result['image']}")
    print(
        f"    不通过 {result['summary']['不通过']} 条，"
        f"无法判断 {result['summary']['无法判断']} 条，"
        f"通过 {result['summary']['通过']} 条"
    )
    for item in result["items"]:
        mark = {"通过": "✓", "不通过": "✗", "无法判断": "?"}[item["status"]]
        print(f"    {mark} {item['id']}  {item['title']}")

    facts = ""
    if args.print_facts:
        from core.context import PageContext

        facts = PageContext.load(args.image, dpi=args.dpi).fact_card()
        print("\n" + facts)

    md_path, json_path = save(result, facts=facts, outdir=args.out or None)
    print(f"\n    反馈文档：{md_path}")
    print(f"    结果数据：{json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
