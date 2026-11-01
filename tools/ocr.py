# -*- coding: utf-8 -*-
"""PaddleOCR 封装：把截图变成结构化的文本行。

上层只调用 recognize(image)，拿到 List[TextLine]，
不关心 PaddleOCR 各版本返回格式的差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from configs import config


@dataclass
class TextLine:
    """一行文字及其外接框（像素坐标，原点在左上角）。"""

    text: str
    box: tuple  # (x1, y1, x2, y2)
    confidence: float = 1.0

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class OCRService:
    """PaddleOCR 的惰性封装：第一次调用才加载模型。"""

    def __init__(self, lang: str = config.OCR_LANG, use_gpu: bool = config.OCR_USE_GPU):
        self.lang = lang
        self.use_gpu = use_gpu
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            self._engine = self._load_paddle()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 paddleocr，请先执行：pip install -r requirements.txt"
            ) from exc
        return self._engine

    def _load_paddle(self) -> Any:
        import sys

        from paddleocr import PaddleOCR

        # 注意：paddleocr 自带的包也叫 tools，导入时会把 sys.modules["tools"]
        # 顶替成它自己的版本。这里把本项目的 tools 包换回来，否则后面
        # `from tools import get_tool` 会去 paddleocr 目录里找。
        ours = sys.modules.get("__pkg_tools__")
        if ours is not None:
            sys.modules["tools"] = ours

        base = {"lang": self.lang, "use_angle_cls": False}
        if config.OCR_MODEL_DIR:
            import os

            os.makedirs(config.OCR_MODEL_DIR, exist_ok=True)
            base.update(
                det_model_dir=os.path.join(config.OCR_MODEL_DIR, "det"),
                rec_model_dir=os.path.join(config.OCR_MODEL_DIR, "rec"),
            )

        # 不同小版本参数名略有差异，逐个尝试
        for kwargs in (
            dict(base, show_log=False, use_gpu=self.use_gpu),
            dict(base, show_log=False),
            base,
        ):
            try:
                return PaddleOCR(**kwargs)
            except (TypeError, ValueError):
                continue
        raise RuntimeError("PaddleOCR 初始化失败，请检查 paddlepaddle 版本")

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        """识别截图中的全部文本行。"""
        engine = self._ensure_engine()
        if hasattr(engine, "predict"):  # PaddleOCR 3.x
            return self._parse_v3(engine.predict(image))
        return self._parse_v2(engine.ocr(image, cls=False))  # PaddleOCR 2.x

    # ------------------------------------------------------------------ #
    def _parse_v2(self, raw: Any) -> List[TextLine]:
        lines: List[TextLine] = []
        for item in _flatten(raw):
            box, text, score = item
            if text and score >= config.OCR_MIN_CONFIDENCE:
                lines.append(TextLine(text=text, box=box, confidence=float(score)))
        return lines

    def _parse_v3(self, raw: Any) -> List[TextLine]:
        lines: List[TextLine] = []
        for page in raw or []:
            if not isinstance(page, dict):
                continue
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                if text and float(score) >= config.OCR_MIN_CONFIDENCE:
                    lines.append(
                        TextLine(text=str(text), box=_poly_to_box(poly), confidence=float(score))
                    )
        return lines


# --------------------------------------------------------------------------- #
# 解析辅助
# --------------------------------------------------------------------------- #
def _flatten(raw: Any):
    """把 2.x 的返回结构统一成 (box, text, score)。"""
    if not raw:
        return
    for item in raw:
        if _looks_like_line(item):
            yield _parse_line(item)
        else:
            for sub in item or []:
                if _looks_like_line(sub):
                    yield _parse_line(sub)


def _looks_like_line(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[1], (list, tuple))
        and len(item[1]) == 2
        and isinstance(item[1][0], str)
    )


def _parse_line(item: Sequence[Any]):
    points, payload = item
    return _poly_to_box(points), payload[0], float(payload[1])


def _poly_to_box(poly: Any) -> tuple:
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    xs, ys = points[:, 0], points[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
