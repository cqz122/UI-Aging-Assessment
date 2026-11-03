# -*- coding: utf-8 -*-
"""页面上下文：把一张截图加工成"事实卡"，供工具和模型使用。

事实卡是本项目最关键的一个中间产物：
    视觉大模型看不到图片，只能看文字。所以要把截图里"看得见的东西"
    尽量转成文字——每行写了什么、多大字号、在什么位置、行距多少。
    转得越准确，模型判断得越准。

事实卡长这样（示例）：
    【页面】demo.png
    【尺寸】1080 × 2340 px，dpi 420
    【文字内容】（逐行，括号内为程序测得的字号）
      1. (y=22) 9:41 [9.1dp]
      2. (y=80) 我的健康 [21.6dp]
    【测量汇总】...
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from configs import config
from tools.ocr import OCRService, TextLine


class PageContext:
    """一张截图的上下文：图片、OCR 文本行、缓存起来的工具结果。"""

    def __init__(
        self,
        image_path: str,
        image: np.ndarray,
        lines: List[TextLine],
        dpi: Optional[float] = None,
        config_module: Any = None,
    ):
        self.image_path = image_path
        self.image = image
        self.lines = lines
        self.dpi = float(dpi if dpi is not None else config.DEVICE_DPI)
        self.config = config_module or config
        self._tools: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, image_path: str, ocr: Optional[OCRService] = None, dpi: Optional[float] = None):
        """读图 + OCR，构造上下文。"""
        image = load_image(image_path)
        service = ocr or OCRService()
        lines = service.recognize(image)
        return cls(image_path, image, lines, dpi=dpi)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def px_per_dp(self) -> float:
        return self.dpi / 160.0

    # ------------------------------------------------------------------ #
    def tool(self, name: str) -> Dict[str, Any]:
        """运行某个测量工具，结果缓存（同一张图只算一次）。"""
        if name not in self._tools:
            from tools import get_tool

            module = get_tool(name)
            self._tools[name] = module.run(self)
        return self._tools[name]

    # ------------------------------------------------------------------ #
    def fact_card(self) -> str:
        """生成给模型看的"页面事实卡"。"""
        rows = self._sorted_lines()
        parts: List[str] = []

        parts.append(f"【页面】{os.path.basename(self.image_path)}")
        parts.append(
            f"【尺寸】{self.width} × {self.height} px，dpi {self.dpi:g}"
            f"（1dp ≈ {self.px_per_dp:.2f}px）"
        )
        parts.append("")
        parts.append("【文字内容】按从上到下排列，圆括号内是该行文字框在截图中的高度")
        if rows:
            for index, line in enumerate(rows, 1):
                parts.append(
                    f"{index:>3}. (y={int(line.y1)}, 高 {line.height:.0f}px) {line.text}"
                )
        else:
            parts.append("（本页没有识别到文字）")

        parts.append("")
        parts.append("【程序测量】")
        parts.extend(self._summary_lines())
        parts.append("")
        parts.append("【说明】文字与数值由 OCR 和像素测量生成；截图随消息一并提供。")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    def _sorted_lines(self) -> List[TextLine]:
        return sorted([l for l in self.lines if l.text.strip()], key=lambda l: (l.y1, l.x1))

    def _summary_lines(self) -> List[str]:
        """把所有工具的结果都列出来，新增工具会自动出现，不用改这里。"""
        from tools import TOOLS

        out: List[str] = []
        for name, module in TOOLS.items():
            try:
                data = self.tool(name)
            except Exception as exc:  # noqa: BLE001 - 单个工具失败不影响整张卡
                out.append(f"- {name}：测量失败（{type(exc).__name__}: {exc}）")
                continue
            detail = "，".join(
                f"{key}={value}"
                for key, value in data.items()
                if isinstance(value, (int, float, str))
            )
            out.append(f"- {module.DESCRIPTION}：{detail}")

        if not out:
            out.append("（没有找到任何测量工具）")

        truncated = [l.text for l in self._sorted_lines() if _looks_truncated(l.text)]
        out.append(
            f"- 疑似被省略号截断的文本行：{len(truncated)} 行"
            + (f"（如「{truncated[0][:16]}」）" if truncated else "")
        )
        return out


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """读取图片，返回 BGR 数组。

    用 imdecode + fromfile 而不是 cv2.imread：Windows 下中文路径会读失败。
    """
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def _looks_truncated(text: str) -> bool:
    """判断一行是否以省略号结尾（被截断）。"""
    return text.strip().endswith(("…", "...", "⋯", ".."))
