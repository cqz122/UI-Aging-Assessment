# -*- coding: utf-8 -*-
"""大模型调用封装（任何 OpenAI 兼容服务）。

支持的输入：
    - 纯文本（事实卡）
    - 文本 + 截图（deepseek-flash 支持视觉，见 api-docs.deepseek.com/guides/vision）

为什么单独一层：
    1. 统一处理图片编码、重试、超时、JSON 解析，业务代码只关心"问什么、拿回什么"；
    2. 换模型（deepseek-flash / deepseek-chat / deepseek-reasoner）只改配置；
    3. 模型不支持图片时自动退化成纯文本，不让整份报告崩掉。

请求形态（OpenAI 兼容）：
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]},
    ]
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

from configs import config


class LLMError(RuntimeError):
    """调用失败时抛出，带可读原因。"""


class LLMClient:
    """对 openai SDK 的薄封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or config.VLM_API_KEY
        self.base_url = base_url or config.VLM_BASE_URL
        self.model = model or config.VLM_MODEL
        self._client: Any = None

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError(
                "未配置大模型 API Key。请在项目根目录的 .env 里写入：\n"
                "    VLM_API_KEY=你的Key"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("未安装 openai，请执行：pip install -r requirements.txt") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=config.VLM_TIMEOUT,
            max_retries=0,  # 重试逻辑自己控制，方便打印
        )
        return self._client

    # ------------------------------------------------------------------ #
    def chat(
        self,
        system: str,
        user: str,
        image_path: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """发一次对话，返回模型输出的纯文本。

        image_path 不为空且模型支持图片时，会把截图一起发过去；
        如果接口报错（例如换成了只支持文本的模型），自动去掉图片重试一次。
        """
        try:
            return self._chat_once(system, user, image_path, temperature, with_image=True)
        except Exception as exc:  # noqa: BLE001
            if image_path and config.SEND_IMAGE:
                # 大概率是模型不支持图片，退化成纯文本再试
                try:
                    return self._chat_once(system, user, None, temperature, with_image=False)
                except Exception:  # noqa: BLE001
                    raise LLMError(f"调用大模型失败：{exc}") from exc
            raise LLMError(f"调用大模型失败：{exc}") from exc

    def _chat_once(
        self,
        system: str,
        user: str,
        image_path: Optional[str],
        temperature: Optional[float],
        with_image: bool,
    ) -> str:
        client = self._ensure_client()

        if with_image and image_path and config.SEND_IMAGE:
            content: Any = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_data_url(image_path),
                        "detail": config.IMAGE_DETAIL,
                    },
                },
            ]
        else:
            content = user

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": config.VLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": config.VLM_MAX_TOKENS,
        }

        last_error: Optional[Exception] = None
        for attempt in range(config.VLM_MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - 网络抖动、限流都重试
                last_error = exc
                if attempt < config.VLM_MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"调用大模型失败：{last_error}")

    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        system: str,
        user: str,
        image_path: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """要求模型返回 JSON 并解析；解析失败会追加一句要求，再试一次。"""
        text = self.chat(system, user, image_path, temperature)
        try:
            return extract_json(text)
        except ValueError:
            fix_prompt = (
                f"{user}\n\n"
                "【提醒】上一条输出不是合法 JSON。请只输出 JSON 对象，"
                "不要任何解释、不要代码块。"
            )
            text = self.chat(system, fix_prompt, image_path, temperature)
            return extract_json(text)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def to_data_url(image_path: str) -> str:
    """把本地图片编码成 data URL。

    大模型按文件内容判断图片格式，不看扩展名，所以这里统一声明 image/png 也不影响。
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出里取出 JSON 对象，容忍 ```json 代码块和前后废话。"""
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON：{text[:120]}")
    return json.loads(cleaned[start:end + 1])
