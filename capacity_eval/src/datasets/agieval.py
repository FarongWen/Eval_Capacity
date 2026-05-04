from __future__ import annotations
import json
from pathlib import Path
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class AGIEvalDataset(BaseBenchmarkDataset):
    """AGIEval MCQ dataset loader. Only keeps multiple-choice tasks, excludes cloze/fill-in-the-blank."""

    MCQ_INDICATORS = {"A", "B", "C", "D", "E", "F", "G", "H"}
    CLOZE_INDICATORS = {"fill", "cloze", "blank", "math", "open"}

    def __init__(self, data_dir: str, split: str = "test"):
        super().__init__(data_dir, "agieval", split)

    def load(self) -> None:
        data_path = Path(self.data_dir)
        items = []
        json_files = sorted(data_path.rglob("*.json"))

        for jf in json_files:
            fname = jf.stem.lower()
            if any(c in fname for c in self.CLOZE_INDICATORS):
                logger.info(f"AGIEval: skipping cloze/non-MCQ file: {jf.name}")
                continue

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for rec in data:
                    item = self._parse_record(rec)
                    if item:
                        items.append(item)
            elif isinstance(data, dict):
                # Some AGIEval files have a top-level key
                for key, records in data.items():
                    if isinstance(records, list):
                        for rec in records:
                            item = self._parse_record(rec)
                            if item:
                                items.append(item)

        self._items = items
        logger.info(f"AGIEval: loaded {len(items)} MCQ items from {self.data_dir}")

    def _parse_record(self, rec: dict) -> MCQItem | None:
        qid = str(rec.get("id", rec.get("question_id", "")))
        question = str(rec.get("question", rec.get("prompt", "")))
        options = rec.get("options", rec.get("choices", {}))
        answer = str(rec.get("answer", rec.get("answerKey", "")))

        # Normalize options
        choices: dict[str, str] = {}
        if isinstance(options, dict):
            for k, v in options.items():
                letter = str(k).strip().upper()
                if letter in self.MCQ_INDICATORS:
                    choices[letter] = str(v)
        elif isinstance(options, list):
            for i, opt in enumerate(options):
                letter = chr(ord("A") + i)
                choices[letter] = str(opt)

        if not choices:
            return None

        # Skip multi-answer
        if len(answer) > 1 and answer.isalpha():
            return None

        # Map numeric answer to letter if needed
        if answer.isdigit() and int(answer) < len(choices):
            answer = chr(ord("A") + int(answer))

        category = str(rec.get("category", rec.get("subject", "")))

        return MCQItem(
            id=qid,
            benchmark="agieval",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=None,
            metadata={"category": category},
        )
