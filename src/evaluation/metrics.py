from __future__ import annotations
import pandas as pd
from scipy import stats
from capacity_eval.src.io_utils import read_jsonl, ensure_dir
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def compute_correlations(
    reference_scores: dict[str, float],
    subset_scores: dict[str, float],
) -> dict[str, float]:
    """Compute SRCC, PLCC, KRCC between reference and subset model scores.

    Both inputs: {model_name: score}
    """
    common_models = sorted(set(reference_scores.keys()) & set(subset_scores.keys()))
    if len(common_models) < 3:
        logger.warning(f"Too few common models ({len(common_models)}) for correlation")
        return {"SRCC": float("nan"), "PLCC": float("nan"), "KRCC": float("nan")}

    ref = [reference_scores[m] for m in common_models]
    sub = [subset_scores[m] for m in common_models]

    srcc, _ = stats.spearmanr(ref, sub)
    plcc, _ = stats.pearsonr(ref, sub)
    krcc, _ = stats.kendalltau(ref, sub)

    return {"SRCC": srcc, "PLCC": plcc, "KRCC": krcc}


def compute_ranking_fidelity(
    output_dir: str,
    benchmark: str,
    reference_scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute ranking fidelity for all subsets of a benchmark."""
    from pathlib import Path
    output_dir = Path(output_dir)

    # If no reference provided, use full benchmark model scores
    if reference_scores is None:
        model_csv = output_dir / "scores" / "model_level_scores.csv"
        if not model_csv.exists():
            logger.error(f"model_level_scores.csv not found at {model_csv}")
            return pd.DataFrame()
        df = pd.read_csv(model_csv)
        ref_df = df[(df["benchmark"] == benchmark) & (df["stage"] == "validation")]
        reference_scores = dict(zip(ref_df["model"], ref_df["accuracy"]))

    results = []
    # Also compute for subset evaluations
    subset_csv = output_dir / "calibration" / benchmark / "capacity_matched_subsets.json"
    if subset_csv.exists():
        from capacity_eval.src.io_utils import read_json
        subset_data = read_json(subset_csv)
        for level in subset_data.get("levels", []):
            for subset in level.get("subsets", []):
                # Compute subset model scores from parsed results
                subset_scores = _get_subset_scores(output_dir, benchmark, subset["item_ids"])
                corr = compute_correlations(reference_scores, subset_scores)
                results.append({
                    "benchmark": benchmark,
                    "lambda": level["lambda"],
                    "subset_type": subset["type"],
                    "k": subset["k"],
                    "C": subset.get("C", float("nan")),
                    **corr,
                })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        fidelity_path = output_dir / "scores" / "ranking_fidelity.csv"
        ensure_dir(output_dir / "scores")
        result_df.to_csv(fidelity_path, index=False)
        logger.info(f"Ranking fidelity saved: {len(result_df)} rows -> {fidelity_path}")

    return result_df


def _get_subset_scores(output_dir, benchmark: str, item_ids: list[str]) -> dict[str, float]:
    """Get model scores on a specific subset of items from parsed results."""
    from pathlib import Path
    parsed_dir = Path(output_dir) / "parsed" / benchmark
    item_id_set = set(item_ids)
    model_scores: dict[str, list[bool]] = {}

    for jsonl_file in parsed_dir.rglob("*.jsonl"):
        records = read_jsonl(jsonl_file)
        for r in records:
            if r.get("item_id") in item_id_set and r.get("stage") == "validation":
                model = r.get("model", "")
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(r.get("correct", False))

    return {m: sum(v) / len(v) for m, v in model_scores.items() if v}
