"""Trim datasets to max 2000 items each. Overwrites original files in-place."""
import random
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

SEED = 42
MAX_ITEMS = 2000


def trim_arrow_file(path: Path) -> None:
    ds = Dataset.from_file(str(path))
    if len(ds) <= MAX_ITEMS:
        print(f"  {path.name}: {len(ds)} items, no trim needed")
        return
    random.seed(SEED)
    indices = random.sample(range(len(ds)), MAX_ITEMS)
    indices.sort()
    trimmed = ds.select(indices)
    # Save back as arrow
    trimmed.save_to_disk(str(path.with_suffix(".trimmed")))
    path.unlink()
    # Arrow save_to_disk creates a directory; move the arrow file back
    trimmed_dir = path.with_suffix(".trimmed")
    arrow_files = list(trimmed_dir.glob("*.arrow"))
    if arrow_files:
        arrow_files[0].rename(path)
    # Clean up the directory
    for f in trimmed_dir.iterdir():
        f.unlink()
    trimmed_dir.rmdir()
    print(f"  {path.name}: {len(ds)} -> {MAX_ITEMS} items")


def trim_parquet_file(path: Path) -> None:
    df = pd.read_parquet(str(path))
    if len(df) <= MAX_ITEMS:
        print(f"  {path.name}: {len(df)} rows, no trim needed")
        return
    random.seed(SEED)
    df = df.sample(n=MAX_ITEMS, random_state=SEED).sort_index()
    df.to_parquet(str(path))
    print(f"  {path.name}: {len(df)} -> {MAX_ITEMS} rows (in-place)")


def trim_jsonl_file(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= MAX_ITEMS:
        print(f"  {path.name}: {len(lines)} lines, no trim needed")
        return
    random.seed(SEED)
    selected = random.sample(lines, MAX_ITEMS)
    selected.sort(key=lambda l: lines.index(l))
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(selected)
    print(f"  {path.name}: {len(lines)} -> {MAX_ITEMS} lines")


def trim_dataset(data_dir: str) -> None:
    root = Path(data_dir)
    print(f"\n--- Trimming {root.name} ---")

    for arrow in sorted(root.rglob("*.arrow")):
        trim_arrow_file(arrow)

    for parquet in sorted(root.rglob("*.parquet")):
        trim_parquet_file(parquet)

    for jsonl in sorted(root.rglob("*.jsonl")):
        trim_jsonl_file(jsonl)


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    for sub in sorted(Path(dataset_dir).iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            trim_dataset(str(sub))
