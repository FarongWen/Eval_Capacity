from __future__ import annotations
import pandas as pd
from pathlib import Path
from capacity_eval.src.io_utils import read_jsonl, ensure_dir
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def compute_scores(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute item-level and model-level scores from parsed results."""
    output_dir = Path(output_dir)
    parsed_dir = output_dir / "parsed"
    all_records = []

    for jsonl_file in parsed_dir.rglob("*.jsonl"):
        all_records.extend(read_jsonl(jsonl_file))

    if not all_records:
        logger.warning("No parsed records found")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Item-level results
    item_df = df[["benchmark", "model", "model_group", "stage", "repeat_id", "item_id", "correct"]].copy()
    item_path = output_dir / "scores" / "item_level_results.csv"
    ensure_dir(item_path.parent)
    item_df.to_csv(item_path, index=False)
    logger.info(f"Item-level results saved: {len(item_df)} rows -> {item_path}")

    # Model-level scores
    model_df = (
        df.groupby(["benchmark", "model", "model_group", "stage"])
        .agg(
            accuracy=("correct", "mean"),
            n_items=("item_id", "count"),
            n_correct=("correct", "sum"),
            n_parsed=("parsed_answer", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    model_path = output_dir / "scores" / "model_level_scores.csv"
    model_df.to_csv(model_path, index=False)
    logger.info(f"Model-level scores saved: {len(model_df)} rows -> {model_path}")

    return item_df, model_df
