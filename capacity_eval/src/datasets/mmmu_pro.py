from __future__ import annotations
import json
import base64
from pathlib import Path
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


class MMMUProDataset(BaseBenchmarkDataset):
    def __init__(self, data_dir: str, split: str = "test"):
        super().__init__(data_dir, "mmmu_pro", split)
        self._images_dir = Path(data_dir) / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        data_path = Path(self.data_dir)
        items = []

        # Try parquet (HuggingFace cache)
        parquet_files = sorted(data_path.rglob("*.parquet"))
        if parquet_files:
            import pandas as pd
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            items = self._parse_dataframe(df)

        self._items = items
        logger.info(f"MMMU-Pro: loaded {len(items)} items from {self.data_dir}")

    def _parse_dataframe(self, df) -> list[MCQItem]:
        items = []
        for idx, row in df.iterrows():
            try:
                item = self._parse_row(idx, row)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Skipping MMMU-Pro row {idx}: {e}")
        return items

    def _parse_row(self, idx, row) -> MCQItem | None:
        import ast
        from PIL import Image
        import io

        qid = str(row.get("id", idx))
        question = str(row.get("question", ""))
        options_raw = row.get("options", "[]")
        if isinstance(options_raw, str):
            options_raw = ast.literal_eval(options_raw)
        answer = str(row.get("answer", ""))

        choices: dict[str, str] = {}
        if isinstance(options_raw, list):
            for i, opt in enumerate(options_raw):
                letter = chr(ord("A") + i)
                choices[letter] = str(opt)
        elif isinstance(options_raw, dict):
            choices = {str(k): str(v) for k, v in options_raw.items()}

        if not choices or not answer:
            return None

        # Handle image extraction - check for image columns
        image_path = None
        for col in ["image", "image_1"]:
            if col in row.index and row[col] is not None:
                img_data = row[col]
                img_filename = f"mmmu_{qid}.jpg"
                img_path = self._images_dir / img_filename

                if not img_path.exists():
                    try:
                        if hasattr(img_data, "save"):
                            img = img_data
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            img.save(str(img_path))
                        elif isinstance(img_data, bytes):
                            img = Image.open(io.BytesIO(img_data))
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            img.save(str(img_path))
                        elif isinstance(img_data, dict) and "bytes" in img_data:
                            img = Image.open(io.BytesIO(img_data["bytes"]))
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            img.save(str(img_path))
                    except Exception as e:
                        logger.warning(f"Could not save image for {qid}: {e}")
                        continue

                image_path = str(img_path)
                break

        subject = str(row.get("subject", ""))

        return MCQItem(
            id=qid,
            benchmark="mmmu_pro",
            split=self.split,
            question=question,
            choices=choices,
            answer=answer,
            image_path=image_path,
            metadata={"subject": subject, "topic_difficulty": str(row.get("topic_difficulty", ""))},
        )
