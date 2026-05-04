from __future__ import annotations
import json
from pathlib import Path
from capacity_eval.src.config import ExperimentConfig
from capacity_eval.src.io_utils import read_json, ensure_dir
from capacity_eval.src.evaluation.metrics import compute_correlations
from capacity_eval.src.evaluation.scorer import compute_scores
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def run_capacity_consistency_experiment(config: ExperimentConfig, group: str) -> None:
    """Experiment 5.3: Single-Observation Capacity-Level Consistency.

    For each benchmark, at each capacity level (lambda),
    verify that min_k and large_k subsets produce similar ranking fidelity.
    """
    benchmarks = config.benchmarks.llm if group == "llm" else config.benchmarks.lmm
    output_dir = Path(config.output_dir)

    # Get model-level reference scores
    from capacity_eval.src.evaluation.scorer import compute_scores
    compute_scores(str(output_dir))

    model_csv = output_dir / "scores" / "model_level_scores.csv"
    import pandas as pd
    if not model_csv.exists():
        logger.error("model_level_scores.csv not found, run evaluation first")
        return

    model_df = pd.read_csv(model_csv)
    results = []

    for benchmark in benchmarks:
        subset_path = output_dir / "calibration" / benchmark / "capacity_matched_subsets.json"
        if not subset_path.exists():
            logger.warning(f"Subset file not found for {benchmark}, run build_subsets first")
            continue

        subset_data = read_json(subset_path)
        C_ref = subset_data["C_ref"]

        # Build reference scores for validation models
        ref_df = model_df[(model_df["benchmark"] == benchmark) & (model_df["stage"] == "validation")]
        ref_scores = dict(zip(ref_df["model"], ref_df["accuracy"]))

        if not ref_scores:
            logger.warning(f"No validation scores for {benchmark}")
            continue

        for level in subset_data.get("levels", []):
            lam = level["lambda"]
            target_C = level["target_C"]
            level_corrs = {}

            for subset in level.get("subsets", []):
                subset_type = subset["type"]
                item_ids = subset["item_ids"]
                k = subset["k"]
                B_eff = subset["B_eff"]
                N = subset["N"]
                S = subset["S"]
                C = subset["C"]

                # Compute subset scores from parsed results
                subset_scores = _compute_subset_model_scores(output_dir, benchmark, item_ids)

                corrs = compute_correlations(ref_scores, subset_scores)

                results.append({
                    "benchmark": benchmark,
                    "lambda": lam,
                    "subset_type": subset_type,
                    "C_ref": C_ref,
                    "target_C": target_C,
                    "C": C,
                    "B_eff": B_eff,
                    "S": S,
                    "N": N,
                    "k": k,
                    "SRCC": corrs["SRCC"],
                    "PLCC": corrs["PLCC"],
                    "KRCC": corrs["KRCC"],
                })
                level_corrs[subset_type] = corrs["SRCC"]

            # Within-level delta SRCC
            if "min_k" in level_corrs and "large_k" in level_corrs:
                delta_srcc = abs(level_corrs["min_k"] - level_corrs["large_k"])
                # Add delta to the last two results
                results[-1]["within_level_delta_SRCC"] = delta_srcc
                results[-2]["within_level_delta_SRCC"] = delta_srcc

    result_df = pd.DataFrame(results)
    out_path = output_dir / "scores" / "exp53_capacity_consistency.csv"
    ensure_dir(out_path.parent)
    result_df.to_csv(out_path, index=False)
    logger.info(f"Exp 5.3 results saved: {len(result_df)} rows -> {out_path}")


def _compute_subset_model_scores(output_dir: Path, benchmark: str, item_ids: list[str]) -> dict[str, float]:
    """Compute validation model accuracy on a subset of items."""
    from capacity_eval.src.io_utils import read_jsonl

    parsed_dir = output_dir / "parsed" / benchmark
    item_id_set = set(item_ids)
    model_scores: dict[str, list[bool]] = {}

    for jsonl_file in parsed_dir.rglob("*.jsonl"):
        records = read_jsonl(jsonl_file)
        for r in records:
            if r.get("stage") == "validation" and r.get("item_id") in item_id_set:
                model = r.get("model", "")
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(r.get("correct", False))

    return {m: sum(v) / len(v) for m, v in model_scores.items() if v}
