from __future__ import annotations
import numpy as np
from scipy import stats
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def estimate_channel_parameters(
    correctness_tensor: dict,
    epsilon: float = 1e-8,
) -> dict:
    """Estimate channel parameters from calibration correctness tensor.

    Returns dict with B_eff, S_cal, N for the full benchmark.
    """
    from capacity_eval.src.calibration.capacity import estimate_B_eff, estimate_S_cal, compute_capacity

    # Collect r_pb per item
    all_item_ids: set[str] = set()
    for model, repeats in correctness_tensor.items():
        for repeat_id, items in repeats.items():
            all_item_ids.update(items.keys())

    # Compute item discrimination
    r_pb_values = []
    for item_id in all_item_ids:
        r_pb = _point_biserial_for_item(correctness_tensor, item_id)
        r_pb_values.append(r_pb)

    B_eff = estimate_B_eff(r_pb_values)

    # Model scores
    model_scores: dict[str, list[float]] = {}
    for model, repeats in correctness_tensor.items():
        for repeat_id, items in repeats.items():
            if items:
                score = np.mean(list(items.values()))
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

    S_cal = estimate_S_cal(model_scores, epsilon)
    from capacity_eval.src.calibration.capacity import estimate_N_subset
    N = estimate_N_subset(correctness_tensor, list(all_item_ids), epsilon)
    C = compute_capacity(B_eff, S_cal, N)

    return {"B_eff": B_eff, "S_cal": S_cal, "N": N, "C": C, "n_items": len(all_item_ids)}


def _point_biserial_for_item(correctness_tensor: dict, item_id: str) -> float:
    """Quick point-biserial for a single item."""
    model_item: dict[str, float] = {}
    model_total: dict[str, float] = {}

    for model, repeats in correctness_tensor.items():
        item_vals = []
        total_vals = []
        for repeat_id, items in repeats.items():
            if item_id in items:
                item_vals.append(float(items[item_id]))
                others = [float(v) for k, v in items.items() if k != item_id]
                if others:
                    total_vals.append(np.mean(others))
        if item_vals:
            model_item[model] = np.mean(item_vals)
            if total_vals:
                model_total[model] = np.mean(total_vals)

    if len(model_item) < 3:
        return 0.0

    models = sorted(model_item.keys())
    a = np.array([model_item[m] for m in models])
    b = np.array([model_total.get(m, 0) for m in models])
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr, _ = stats.pearsonr(a, b)
    return corr if not np.isnan(corr) else 0.0
