from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from capacity_eval.src.io_utils import read_jsonl, write_json, ensure_dir, read_json
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def compute_item_statistics(
    output_dir: str,
    benchmark: str,
    epsilon: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """Compute item-level statistics from calibration repeated results.

    Returns:
        item_statistics DataFrame and correctness_tensor dict
    """
    from pathlib import Path
    parsed_dir = Path(output_dir) / "parsed" / benchmark

    # Build correctness tensor: {model: {repeat: {item: bool}}}
    correctness_tensor: dict[str, dict[int, dict[str, bool]]] = {}
    all_item_ids: set[str] = set()

    for jsonl_file in parsed_dir.rglob("*.jsonl"):
        records = read_jsonl(jsonl_file)
        for r in records:
            if r.get("stage") != "calibration":
                continue
            model = r["model"]
            repeat_id = r.get("repeat_id", 0)
            item_id = r.get("item_id", "")
            correct = r.get("correct", False)
            if model not in correctness_tensor:
                correctness_tensor[model] = {}
            if repeat_id not in correctness_tensor[model]:
                correctness_tensor[model][repeat_id] = {}
            correctness_tensor[model][repeat_id][item_id] = correct
            all_item_ids.add(item_id)

    if not all_item_ids:
        logger.warning(f"No calibration data found for {benchmark}")
        return pd.DataFrame(), correctness_tensor

    # Compute model-level scores per repeat for S_cal
    model_scores: dict[str, list[float]] = {}
    for model, repeats in correctness_tensor.items():
        for repeat_id, items in repeats.items():
            if items:
                score = np.mean(list(items.values()))
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

    # Compute per-item statistics
    item_stats = []
    for item_id in sorted(all_item_ids):
        # Collect all (model, repeat) correctness for this item
        item_correctness: dict[str, list[bool]] = {}  # model -> [bool, bool, ...]
        for model, repeats in correctness_tensor.items():
            vals = []
            for repeat_id, items in repeats.items():
                if item_id in items:
                    vals.append(items[item_id])
            if vals:
                item_correctness[model] = vals

        if not item_correctness:
            continue

        # p_i = overall accuracy
        all_vals = [v for vals in item_correctness.values() for v in vals]
        p_i = np.mean(all_vals) if all_vals else 0.0

        # item variance
        var_i = np.var(all_vals, ddof=1) if len(all_vals) >= 2 else 0.0

        # item noise: n_i = mean_m Var_r(x_{m,i,r})
        model_vars = []
        for model, vals in item_correctness.items():
            if len(vals) >= 2:
                model_vars.append(np.var(vals, ddof=1))
        n_i = np.mean(model_vars) if model_vars else epsilon

        # Point-biserial correlation r_pb,i
        # For each repeat, compute model mean score and item correctness
        r_pb = _compute_point_biserial(correctness_tensor, item_id)

        item_stats.append({
            "item_id": item_id,
            "p_i": p_i,
            "var_i": var_i,
            "n_i": n_i,
            "r_pb": r_pb,
        })

    item_df = pd.DataFrame(item_stats)

    # Save
    stats_path = Path(output_dir) / "calibration" / benchmark / "item_statistics.csv"
    ensure_dir(stats_path.parent)
    item_df.to_csv(stats_path, index=False)
    logger.info(f"Item statistics saved: {len(item_df)} items -> {stats_path}")

    # Save full capacity for the benchmark
    from capacity_eval.src.calibration.capacity import estimate_B_eff, estimate_S_cal, compute_capacity

    r_pb_values = item_df["r_pb"].dropna().tolist()
    B_eff = estimate_B_eff(r_pb_values)
    S_cal = estimate_S_cal(model_scores, epsilon=epsilon)
    N_full = _compute_N_full(correctness_tensor, list(all_item_ids), epsilon)

    C_ref = compute_capacity(B_eff, S_cal, N_full)

    capacity_data = {
        "benchmark": benchmark,
        "B_eff": B_eff,
        "S_cal": S_cal,
        "N": N_full,
        "C_ref": C_ref,
        "n_items": len(all_item_ids),
    }
    cap_path = Path(output_dir) / "calibration" / benchmark / "capacity_full.json"
    write_json(cap_path, capacity_data)
    logger.info(f"Full benchmark capacity: B_eff={B_eff:.4f}, S={S_cal:.4f}, N={N_full:.6f}, C={C_ref:.4f}")

    return item_df, correctness_tensor


def _compute_point_biserial(
    correctness_tensor: dict[str, dict[int, dict[str, bool]]],
    item_id: str,
) -> float:
    """Compute point-biserial discrimination for an item.

    For each model, compute mean score across all items (excluding this one),
    then correlate with this item's correctness per model.
    """
    # Average item correctness per model (across repeats)
    model_item_correct: dict[str, float] = {}
    model_total_score: dict[str, float] = {}

    for model, repeats in correctness_tensor.items():
        item_vals = []
        total_vals = []
        for repeat_id, items in repeats.items():
            if item_id in items:
                item_vals.append(float(items[item_id]))
                # Model total score on this repeat (excluding this item)
                other = [float(v) for k, v in items.items() if k != item_id]
                if other:
                    total_vals.append(np.mean(other))

        if item_vals:
            model_item_correct[model] = np.mean(item_vals)
            if total_vals:
                model_total_score[model] = np.mean(total_vals)

    if len(model_item_correct) < 3:
        return 0.0

    # Point-biserial: correlate item correctness (binary) with total score
    models = sorted(model_item_correct.keys())
    item_arr = np.array([model_item_correct[m] for m in models])
    total_arr = np.array([model_total_score.get(m, 0.0) for m in models])

    if np.std(item_arr) == 0 or np.std(total_arr) == 0:
        return 0.0

    corr, _ = stats.pearsonr(item_arr, total_arr)
    return corr if not np.isnan(corr) else 0.0


def _compute_N_full(
    correctness_tensor: dict, all_item_ids: list[str], epsilon: float
) -> float:
    from capacity_eval.src.calibration.capacity import estimate_N_subset
    return estimate_N_subset(correctness_tensor, all_item_ids, epsilon)
