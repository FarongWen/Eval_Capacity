from __future__ import annotations
import csv
import base64
from pathlib import Path
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()

csv.field_size_limit(100 * 1024 * 1024)


class MMBenchDataset(BaseBenchmarkDataset):
    def __init__(self, data_dir: str, split: str = "test"):
        super().__init__(data_dir, "mmbench", split)
        self._images_dir = Path(data_dir) / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        data_path = Path(self.data_dir)
        items = []

        # Try tsv first (MMBench default format)
        tsv_files = sorted(data_path.rglob("*.tsv"))
        if tsv_files:
            for tf in tsv_files:
                items.extend(self._load_tsv(tf))
        else:
            # Fallback to csv/parquet
            csv_files = sorted(data_path.rglob("*.csv"))
            for cf in csv_files:
                items.extend(self._load_csv(cf))
            parquet_files = sorted(data_path.rglob("*.parquet"))
            if parquet_files:
                import pandas as pd
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                items.extend(self._parse_dataframe(df))

        self._items = items
        logger.info(f"MMBench: loaded {len(items)} items from {self.data_dir}")

    def _load_tsv(self, path: Path) -> list[MCQItem]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                item = self._parse_dict_row(row)
                if item:
                    items.append(item)
        return items

    def _load_csv(self, path: Path) -> list[MCQItem]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = self._parse_dict_row(row)
                if item:
                    items.append(item)
        return items

    def _parse_dataframe(self, df) -> list[MCQItem]:
        items = []
        for _, row in df.iterrows():
            item = self._parse_dict_row(row.to_dict())
            if item:
                items.append(item)
        return items

    def _parse_dict_row(self, row: dict) -> MCQItem | None:
        qid = str(row.get("index", row.get("id", row.get("question_id", ""))))
        question = str(row.get("question", ""))

        # MMBench typically has A/B/C/D columns or an options field
        choices: dict[str, str] = {}
        for letter in ["A", "B", "C", "D", "E", "F"]:
            col = letter
            if col in row and row[col] and str(row[col]).strip():
                choices[letter] = str(row[col]).strip()

        if not choices and "options" in row:
            import ast
            opts = row["options"]
            if isinstance(opts, str):
                try:
                    opts = ast.literal_eval(opts)
                except Exception:
                    pass
            if isinstance(opts, (list, tuple)):
                for i, opt in enumerate(opts):
                    letter = chr(ord("A") + i)
                    choices[letter] = str(opt)

        answer = str(row.get("answer", row.get("target", "")))

        if not choices or not answer:
            return None

        # Handle image
        image_path = None
        img_data = row.get("image", row.get("image_base64", None))
        if img_data and str(img_data).strip():
            img_filename = f"mmbench_{qid}.jpg"
            img_path = self._images_dir / img_filename
            if not img_path.exists():
                try:
                    b64 = str(img_data)
                    if b64.startswith("data:image"):
                        b64 = b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                except Exception as e:
                    logger.warning(f"Could not decode image for {qid}: {e}")
            if img_path.exists():
                image_path = str(img_path)

        # If image_path column exists directly
        if not image_path and "image_path" in row and row["image_path"]:
            image_path = str(row["image_path"])

        return MCQItem(
            id=qid,
            benchmark="mmbench",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=image_path,
            metadata={"source": str(row.get("source", ""))},
        )
