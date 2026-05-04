from __future__ import annotations
from typing import Optional
from capacity_eval.src.models.base import BaseModelClient
from capacity_eval.src.config import ModelConfig
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class LocalModelClient(BaseModelClient):
    """Placeholder for local model inference via vLLM or transformers.

    Interface is consistent with APIModelClient. Actual implementation depends
    on deployment. TODO: implement vLLM / transformers backend.
    """

    def __init__(self, config: ModelConfig):
        self._config = config
        # TODO: initialize vLLM or transformers pipeline
        logger.warning(f"LocalModelClient for {config.name} is a placeholder — not yet functional")

    @property
    def supports_vision(self) -> bool:
        return self._config.supports_vision

    @property
    def model_name(self) -> str:
        return self._config.name

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        raise NotImplementedError(
            f"Local inference for {self._config.name} is not implemented. "
            "Deploy the model with vLLM and use APIModelClient with base_url pointing to it."
        )
