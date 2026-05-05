from __future__ import annotations
import json
from pathlib import Path
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class AGIEvalDataset(BaseBenchmarkDataset):
    """AGIEval MCQ dataset loader. Reads from merged agieval_mcq.jsonl."""

    def __init__(self, data_dir: str, split: str = "test"):
        super().__init__(data_dir, "agieval", split)

    def load(self) -> None:
        data_path = Path(self.data_dir)
        items = []

        # Prefer merged file
        merged = data_path / "agieval_mcq.jsonl"
        if merged.exists():
            items = self._load_jsonl(merged)
        else:
            # Fallback: scan individual jsonl files
            for jf in sorted(data_path.rglob("*.jsonl")):
                items.extend(self._load_jsonl(jf))

        self._items = items
        logger.info(f"AGIEval: loaded {len(items)} MCQ items from {self.data_dir}")

    def _load_jsonl(self, path: Path) -> list[MCQItem]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                item = self._parse_record(rec, i, path.stem)
                if item:
                    items.append(item)
        return items

    def _parse_record(self, rec: dict, line_idx: int, source: str) -> MCQItem | None:
        # AGIEval fields: passage, question, options, label, answer, other
        question = str(rec.get("question", ""))
        passage = rec.get("passage")
        if passage and str(passage).strip() and str(passage) != "None":
            question = f"{passage}\n{question}"

        options = rec.get("options")
        label = rec.get("label")

        if not options or not label:
            return None

        choices: dict[str, str] = {}
        if isinstance(options, list):
            for i, opt in enumerate(options):
                letter = chr(ord("A") + i)
                choices[letter] = str(opt)
        elif isinstance(options, dict):
            for k, v in options.items():
                letter = str(k).strip().upper()
                if len(letter) == 1 and letter.isalpha():
                    choices[letter] = str(v)

        if not choices:
            return None

        answer = str(label).strip().upper()
        if len(answer) != 1 or not answer.isalpha():
            return None

        qid = f"agieval_{source}_{line_idx}"
        src = rec.get("_source", source)

        return MCQItem(
            id=qid,
            benchmark="agieval",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=None,
            metadata={"source": src},
        )
