# -*- coding: utf-8 -*-
"""对比度测量工具（本项目唯一的参考实现）。

照着这个文件的结构写新工具即可，一个新工具只需要三样东西：
    NAME         工具名，条目里用它引用
    DESCRIPTION  一句话说明，`python main.py --list-tools` 会显示
    run(page)    输入 PageContext，输出一个字典

输入 page 里有：
    page.image   截图（numpy 数组，BGR）
    page.lines   OCR 结果，每行有 .text / .x1 .y1 .x2 .y2 / .width / .height
    page.dpi     设备像素密度
    page.config  阈值常量

输出字典会被用于三处：条目的 judge 取值、写进页面事实卡给模型看、存进结果 JSON。

原理：
    对每行文字裁出一个小块，用 Otsu 阈值把像素分成“深色组”和“浅色组”，
    像素少的一组通常是文字笔画，另一组是背景；再按 WCAG 2.1 算
        ratio = (L_light + 0.05) / (L_dark + 0.05)
"""

from __future__ import annotations

from typing import Any, Dict, List

NAME = "contrast"
DESCRIPTION = "测量文字与背景的对比度（WCAG 2.1）"

# run() 返回一个字典，每个字段都有明确用途：
#   mean_ratio  平均对比度：一眼看整体水平，也会写进事实卡给模型参考
#   min_ratio   最低对比度：指出"哪几行最看不清"，报告里当证据
#   line_count  参与统计的行数：判断这次测量可不可信（为 0 说明页面上没文字）
#   low_count   低于阈值的行数：判定用，直接对应"有几行不合格"
#   threshold   本次使用的阈值：写进报告，别人复核时能对上口径
#
# 为什么要这么多字段，而不是只返回一个数字：
#   判定要能落到"哪几行不合格"，光有平均值会把个别行的问题平均掉；
#   有了 low_count 才能判 A03，有了 min_ratio 才能说清最差的那行差多少，
#   有了 line_count 才能在页面上没文字时判"无法判断"，而不是误判通过。


def run(page: Any) -> Dict[str, Any]:
    ratios: List[tuple] = []
    for line in page.lines:
        if not line.text.strip():
            continue
        patch = _crop(page.image, line)
        if patch is None:
            continue
        ratios.append((_contrast(patch), max(len(line.text), 1)))

    if not ratios:
        return {"mean_ratio": 0.0, "min_ratio": 0.0, "line_count": 0,
                "low_count": 0, "threshold": page.config.CONTRAST_MIN}

    threshold = page.config.CONTRAST_MIN
    # 按字符数加权：长文本更能代表“这一屏的整体可读性”
    total = sum(weight for _, weight in ratios)
    mean = sum(ratio * weight for ratio, weight in ratios) / total

    return {
        "mean_ratio": round(mean, 2),
        "min_ratio": round(min(r for r, _ in ratios), 2),
        "line_count": len(ratios),
        "low_count": sum(1 for r, _ in ratios if r < threshold),
        "threshold": threshold,
    }


def _crop(image, line):
    """按文本行裁出小图，四周各留 2px 背景。"""
    height, width = image.shape[:2]
    x1, y1 = max(0, int(line.x1) - 2), max(0, int(line.y1) - 2)
    x2, y2 = min(width, int(line.x2) + 2), min(height, int(line.y2) + 2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return image[y1:y2, x1:x2]


def _contrast(patch) -> float:
    """一小块图像的前景/背景对比度。"""
    import cv2

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray <= threshold
    if dark.sum() == 0 or dark.sum() == dark.size:
        return 1.0  # 纯色块，没有文字

    # 文字笔画只占少数像素，用数量判断谁是前景
    dark_is_text = dark.mean() <= 0.5
    fg = patch[dark].mean(axis=0) if dark_is_text else patch[~dark].mean(axis=0)
    bg = patch[~dark].mean(axis=0) if dark_is_text else patch[dark].mean(axis=0)

    high, low = sorted([_luminance(fg), _luminance(bg)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def _luminance(bgr) -> float:
    """WCAG 相对亮度：sRGB 线性化后按人眼敏感度加权。"""
    b, g, r = (float(c) / 255.0 for c in bgr[:3])

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
