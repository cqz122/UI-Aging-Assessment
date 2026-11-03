# -*- coding: utf-8 -*-
"""本地测量工具包：自动扫描并登记 tools/ 下的所有工具。

==================== 新增一个工具要做的事 ====================
在 tools/ 下新建一个 .py 文件，文件里写下面三样东西即可：

    NAME = "contrast"                                  # 工具名，条目里用它引用
    DESCRIPTION = "测量文字与背景的对比度（WCAG 2.1）"   # 一句话说明
    def run(page) -> dict:                             # 输入 PageContext，输出字典
        ...

不需要改这个文件，也不需要改任何别人的文件 —— 程序启动时自动发现。
（文件名以 _ 开头的不加载，可以放草稿。）
==========================================================

输入 page 是 core/context.py 里的 PageContext：
    page.image   截图（numpy 数组，BGR）
    page.lines   OCR 结果，每行有 text / box / x1..y2 / width / height
    page.dpi     设备像素密度，dp = px * 160 / dpi
    page.config  阈值常量，如 page.config.CONTRAST_MIN
    page.tool("font_size")  复用别的工具结果（自动缓存）

返回的字典会：① 供条目的 judge 取值；② 写进页面事实卡；③ 存进结果 JSON。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

# paddleocr 内部也有一个叫 tools 的包，导入时会把 sys.modules["tools"] 顶掉。
# 这里给自己的包留一个别名，ocr.py 在导入 paddleocr 之后据此换回来。
sys.modules.setdefault("__pkg_tools__", sys.modules[__name__])

# 工具表：名字 -> 模块（启动时自动收集）
TOOLS: Dict[str, Any] = {}


def _discover() -> None:
    """扫描本目录下所有 .py，把带有 NAME 和 run() 的模块收集成工具。"""
    folder = Path(__file__).parent
    for path in sorted(folder.glob("*.py")):
        stem = path.stem
        if stem.startswith("_") or stem == "__init__":
            continue
        module = importlib.import_module(f"tools.{stem}")
        if hasattr(module, "NAME") and callable(getattr(module, "run", None)):
            TOOLS[module.NAME] = module


_discover()


def get_tool(name: str):
    """按名字取工具模块，名字写错时给出可读提示。"""
    if name not in TOOLS:
        available = "、".join(sorted(TOOLS)) or "（空）"
        raise KeyError(f"没有名为 “{name}” 的工具。可用的工具：{available}")
    return TOOLS[name]


def list_tools() -> List[Dict[str, str]]:
    """列出所有工具，供命令行展示。"""
    return [
        {"name": module.NAME, "description": module.DESCRIPTION}
        for module in TOOLS.values()
    ]
