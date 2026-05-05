from __future__ import annotations
import json
from pathlib import Path
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class MMLUProDataset(BaseBenchmarkDataset):
    def __init__(self, data_dir: str, split: str = "test"):
        super().__init__(data_dir, "mmlu_pro", split)

    def load(self) -> None:
        data_path = Path(self.data_dir)
        items = []

        # Try loading via HuggingFace datasets (handles arrow/parquet cache)
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(data_path))
            items = self._parse_hf_dataset(ds)
        except Exception:
            pass

        if not items:
            # Try loading from parquet files (HuggingFace cache format)
            parquet_files = sorted(data_path.rglob("*.parquet"))
            if parquet_files:
                import pandas as pd
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                items = self._parse_dataframe(df)

        if not items:
            # Try arrow files via datasets library
            arrow_files = sorted(data_path.rglob("*.arrow"))
            if arrow_files:
                for af in arrow_files:
                    try:
                        import pandas as pd
                        from datasets import Dataset
                        ds_split = Dataset.from_file(str(af))
                        df = ds_split.to_pandas()
                        items.extend(self._parse_dataframe(df))
                    except Exception as e:
                        logger.warning(f"Failed to load arrow file {af}: {e}")

        if not items:
            # Fallback: csv
            csv_files = sorted(data_path.rglob("*.csv"))
            if csv_files:
                import pandas as pd
                dfs = [pd.read_csv(f) for f in csv_files]
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                items = self._parse_dataframe(df)

        if not items:
            # Fallback: json/jsonl
            json_files = sorted(data_path.rglob("*.json"))
            jsonl_files = sorted(data_path.rglob("*.jsonl"))
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    items.extend(self._parse_record(r) for r in data)
            for jf in jsonl_files:
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(self._parse_record(json.loads(line)))

        self._items = items
        logger.info(f"MMLU-Pro: loaded {len(items)} items from {self.data_dir}")

    def _parse_hf_dataset(self, ds) -> list[MCQItem]:
        """Parse a HuggingFace Dataset object (from load_from_disk)."""
        items = []
        df = ds.to_pandas() if hasattr(ds, 'to_pandas') else ds
        if hasattr(df, 'iterrows'):
            items = self._parse_dataframe(df)
        return items

    def _parse_dataframe(self, df) -> list[MCQItem]:
        items = []
        for _, row in df.iterrows():
            try:
                item = self._parse_row(row)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Skipping MMLU-Pro row: {e}")
        return items

    def _parse_row(self, row) -> MCQItem | None:
        import ast
        # MMLU-Pro fields: question_id, question, options, answer, answer_index, cot_content, category, src
        qid = str(row.get("question_id", row.get("id", "")))
        question = str(row.get("question", ""))
        options_raw = row.get("options", "[]")
        if isinstance(options_raw, str):
            options_raw = ast.literal_eval(options_raw)
        answer = str(row.get("answer", ""))
        category = str(row.get("category", ""))

        choices = {}
        for i, opt in enumerate(options_raw):
            letter = chr(ord("A") + i)
            choices[letter] = str(opt)

        if not choices or not answer:
            return None

        # Skip multi-answer items
        if len(answer) > 1 and answer.isalpha():
            logger.debug(f"Multi-answer item {qid}: {answer}, skipping")
            return None

        return MCQItem(
            id=qid,
            benchmark="mmlu_pro",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=None,
            metadata={"category": category, "src": str(row.get("src", ""))},
        )

    def _parse_record(self, rec: dict) -> MCQItem:
        import ast
        qid = str(rec.get("question_id", rec.get("id", "")))
        question = str(rec.get("question", ""))
        options_raw = rec.get("options", "[]")
        if isinstance(options_raw, str):
            options_raw = ast.literal_eval(options_raw)
        answer = str(rec.get("answer", ""))
        choices = {}
        for i, opt in enumerate(options_raw):
            letter = chr(ord("A") + i)
            choices[letter] = str(opt)
        return MCQItem(
            id=qid,
            benchmark="mmlu_pro",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=None,
            metadata={"category": str(rec.get("category", "")), "src": str(rec.get("src", ""))},
        )
