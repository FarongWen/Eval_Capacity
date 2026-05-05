from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class BaseModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        ...

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
