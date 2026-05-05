from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator
from abc import ABC, abstractmethod


@dataclass
class MCQItem:
    id: str
    benchmark: str
    split: str
    question: str
    choices: dict[str, str]
    answer: str
    image_path: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmarkDataset(ABC):
    def __init__(self, data_dir: str, benchmark_name: str, split: str = "test"):
        self.data_dir = data_dir
        self.benchmark_name = benchmark_name
        self.split = split
        self._items: list[MCQItem] = []

    @abstractmethod
    def load(self) -> None:
        ...

    def iter_items(self) -> Iterator[MCQItem]:
        return iter(self._items)

    def get_all_items(self) -> list[MCQItem]:
        return self._items

    def get_item_by_id(self, item_id: str) -> Optional[MCQItem]:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def __len__(self) -> int:
        return len(self._items)
