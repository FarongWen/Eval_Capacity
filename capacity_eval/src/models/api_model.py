from __future__ import annotations
import os
import time
import base64
import logging
from typing import Optional
from openai import OpenAI
from capacity_eval.src.models.base import BaseModelClient
from capacity_eval.src.config import ModelConfig
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class APIModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig, max_retries: int = 5, retry_delay: float = 2.0):
        self._config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        api_key = os.environ.get(config.api_key_env, "")
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
        )

    @property
    def supports_vision(self) -> bool:
        return self._config.supports_vision

    @property
    def model_name(self) -> str:
        return self._config.name

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        if image_path and self._config.supports_vision:
            messages = self._build_multimodal_message(prompt, image_path)
        else:
            messages = self._build_text_message(prompt)

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model_id,
                    messages=messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait = self._retry_delay * (2 ** attempt)
                logger.warning(f"[{self._config.name}] API error (attempt {attempt+1}/{self._max_retries}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self._max_retries} retries for model {self._config.name}")

    def _build_text_message(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def _build_multimodal_message(self, prompt: str, image_path: str) -> list[dict]:
        b64 = self._image_to_base64(image_path)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{b64}"},
                },
            ],
        }]

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
