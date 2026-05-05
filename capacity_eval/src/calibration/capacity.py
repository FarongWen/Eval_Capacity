from __future__ import annotations
import numpy as np
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def estimate_B_eff(r_pb_values: list[float]) -> float:
    """Estimate effective bandwidth B_eff as sum of absolute point-biserial discrimination coefficients."""
    return sum(abs(r) for r in r_pb_values)


def estimate_N_subset(correctness_tensor: dict, subset_item_ids: list[str], epsilon: float = 1e-8) -> float:
    """Estimate noise N(A) for a subset A.

    N(A) = mean_m Var_r( mean_{i in A} x_{m,i,r} )

    correctness_tensor: {model_name: {repeat_id: {item_id: bool}}}
    """
    if not subset_item_ids:
        return epsilon

    model_variances = []
    item_id_set = set(subset_item_ids)

    for model_name, repeats in correctness_tensor.items():
        repeat_means = []
        for repeat_id, items in repeats.items():
            scores = [float(items.get(iid, False)) for iid in subset_item_ids if iid in items]
            if scores:
                repeat_means.append(np.mean(scores))
        if len(repeat_means) >= 2:
            model_variances.append(np.var(repeat_means, ddof=1))

    N = np.mean(model_variances) if model_variances else epsilon
    return max(N, epsilon)


def estimate_S_cal(model_scores: dict[str, list[float]], epsilon: float = 1e-8) -> float:
    """Estimate signal S_cal from calibration model scores.

    S = Var_m( mean_r score_{m,r} )
    model_scores: {model_name: [score_repeat1, score_repeat2, ...]}
    """
    model_means = []
    for model_name, scores in model_scores.items():
        if scores:
            model_means.append(np.mean(scores))

    if len(model_means) < 2:
        return epsilon

    S = np.var(model_means, ddof=1)
    return max(S, epsilon)


def compute_capacity(B_eff: float, S: float, N: float) -> float:
    """C = B_eff * log2(1 + S / N)"""
    if N <= 0:
        N = 1e-8
    return B_eff * np.log2(1 + S / N)
