"""Merge AGIEval jsonl files into one, keeping only MCQ items, capped at 2000."""
import json
import random
from pathlib import Path

SEED = 42
MAX_ITEMS = 2000
AGIEVAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dataset" / "agieval"


def is_mcq(item: dict) -> bool:
    opts = item.get("options")
    label = item.get("label")
    return (
        opts is not None
        and label is not None
        and isinstance(opts, list)
        and len(opts) >= 2
        and str(label).strip() != ""
        and len(str(label)) == 1
        and str(label).isalpha()
    )


def main():
    mcq_items = []
    for p in sorted(AGIEVAL_DIR.rglob("*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if is_mcq(item):
                    item["_source"] = p.stem
                    mcq_items.append(item)

    print(f"Total MCQ items: {len(mcq_items)}")

    if len(mcq_items) > MAX_ITEMS:
        random.seed(SEED)
        mcq_items = random.sample(mcq_items, MAX_ITEMS)
        print(f"Trimmed to {MAX_ITEMS}")

    out_path = AGIEVAL_DIR / "agieval_mcq.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in mcq_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
