# -*- coding: utf-8 -*-
"""全局配置：所有可调参数集中在这里，改完直接生效，不需要 YAML。

放在 configs/config.py 而不是 .yaml，是为了让组员少学一种格式：
要改阈值就改这一行常量，要改模型就改 DEEPSEEK_MODEL。
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = BASE_DIR / "prompts"          # 12 条检测条目，一条一个文件
OUTPUT_DIR = BASE_DIR / "outputs"          # 反馈文档与结果 JSON
LABEL_DIR = BASE_DIR / "eval" / "labels"   # 人工标注，用于算召回率


def load_env(path: Path | None = None) -> None:
    """读取项目根目录的 .env（KEY=VALUE 格式），不覆盖已有的系统环境变量。

    自己解析十几行，是为了少装一个 python-dotenv 依赖。
    """
    env_path = path or (BASE_DIR / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env()

# --------------------------------------------------------------------------- #
# 截图设备
# --------------------------------------------------------------------------- #
# 像素密度，用来把 px 换算成与设备无关的 dp：dp = px * 160 / dpi
# 本课题统一用 mumu 模拟器，机型 vivo X90s（V2241A）：1080×1920，DPI 480
# 换设备时改这里，或用 .env 里的 AGEFRIENDLY_DPI 覆盖，避免改代码
DEVICE_DPI = float(os.getenv("AGEFRIENDLY_DPI", "480"))

# --------------------------------------------------------------------------- #
# 检测阈值（对应检测条目的判定口径）
# --------------------------------------------------------------------------- #
# A03：文字与背景的对比度下限（WCAG 2.1 正文标准）
CONTRAST_MIN = 4.5
# A03：大字（>=18dp）可放宽到 3:1，需要时在工具里单独统计
CONTRAST_MIN_LARGE = 3.0
LARGE_TEXT_DP = 18.0

# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
OCR_LANG = "ch"
OCR_USE_GPU = False
OCR_MIN_CONFIDENCE = 0.5
# 模型缓存目录。留空用 ~/.paddleocr；服务器不能写用户目录时改成 "models"
OCR_MODEL_DIR = os.getenv("AGEFRIENDLY_MODEL_DIR", "")

# --------------------------------------------------------------------------- #
# 大模型接口（任何 OpenAI 兼容服务都能接）
# --------------------------------------------------------------------------- #
# 默认用 DeepSeek 的 deepseek-flash（支持图片输入，见 api-docs.deepseek.com/guides/vision）。
# 换成别家只要改 .env 里的三个值，代码不用动：
#   火山方舟/豆包  VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
#                  VLM_MODEL=你的接入点 ID，形如 ep-20250101-xxxxx
#   阿里云百炼     VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#                  VLM_MODEL=qwen-vl-max
# 变量优先读 VLM_*，没有时回退到旧的 DEEPSEEK_*，老配置继续能用。
VLM_API_KEY = os.getenv("VLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
VLM_BASE_URL = os.getenv("VLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
VLM_MODEL = os.getenv("VLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
VLM_TEMPERATURE = 0.0     # 判定类任务用 0，保证同一张图结果稳定
VLM_TIMEOUT = 90
VLM_MAX_RETRIES = 2
VLM_MAX_TOKENS = 1200

# 是否把截图一起发给模型。
# True：模型能"看见"颜色、图标、弹窗、遮挡，判断更准；
# False：只发文字版事实卡，适合不支持图片的模型，或想省 token。
SEND_IMAGE = os.getenv("AGEFRIENDLY_SEND_IMAGE", "1") not in {"0", "false", "False"}
# 图片精度：high=保留原图；low=压到 512×512（便宜但小字会糊）
IMAGE_DETAIL = os.getenv("AGEFRIENDLY_IMAGE_DETAIL", "high")

# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
REPORT_TITLE = "适老化检测反馈"
