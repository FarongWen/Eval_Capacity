from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from capacity_eval.src.io_utils import read_json, write_json, ensure_dir
from capacity_eval.src.calibration.capacity import estimate_B_eff, estimate_N_subset, estimate_S_cal, compute_capacity
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def build_capacity_matched_subsets(
    output_dir: str,
    benchmark: str,
    target_levels: list[float] | None = None,
    tolerance: float = 0.10,
    fallback_tolerance: float = 0.15,
    epsilon: float = 1e-8,
) -> dict:
    """Build capacity-matched subsets for Experiment 5.3.

    For each target level lambda, build min_k and large_k subsets that match
    C ≈ lambda * C_ref.
    """
    if target_levels is None:
        target_levels = [0.1, 0.2, 0.3]

    # Load item statistics and full capacity
    stats_path = Path(output_dir) / "calibration" / benchmark / "item_statistics.csv"
    cap_path = Path(output_dir) / "calibration" / benchmark / "capacity_full.json"
    tensor_path = Path(output_dir) / "calibration" / benchmark / "correctness_tensor.json"

    if not stats_path.exists() or not cap_path.exists():
        logger.error(f"Calibration data not found for {benchmark}")
        return {}

    item_df = pd.read_csv(stats_path)
    cap_data = read_json(cap_path)

    C_ref = cap_data["C_ref"]
    S_cal = cap_data["S_cal"]

    # Load correctness tensor
    correctness_tensor = read_json(tensor_path) if tensor_path.exists() else {}
    # Convert string keys in repeat dicts back to int
    for model in correctness_tensor:
        repeats = correctness_tensor[model]
        new_repeats = {}
        for k, v in repeats.items():
            new_repeats[int(k)] = v
        correctness_tensor[model] = new_repeats

    # Sort items by absolute discrimination (high to low)
    item_df = item_df.sort_values("r_pb", key=lambda x: x.abs(), ascending=False)

    # Save candidate subsets
    candidates_path = Path(output_dir) / "calibration" / benchmark / "subset_candidates.csv"

    levels_output = []

    for lam in target_levels:
        target_C = lam * C_ref
        logger.info(f"Building subsets for lambda={lam}, target_C={target_C:.4f}")

        # Min-k subset: greedily add items by discrimination
        min_k_subset = _find_min_k_subset(
            item_df=item_df,
            correctness_tensor=correctness_tensor,
            target_C=target_C,
            S_cal=S_cal,
            tolerance=tolerance,
            fallback_tolerance=fallback_tolerance,
            epsilon=epsilon,
        )

        # Large-k subset: add more items but keep C close to target
        large_k_subset = _find_large_k_subset(
            item_df=item_df,
            correctness_tensor=correctness_tensor,
            target_C=target_C,
            S_cal=S_cal,
            min_k=len(min_k_subset["item_ids"]),
            tolerance=tolerance,
            fallback_tolerance=fallback_tolerance,
            epsilon=epsilon,
        )

        levels_output.append({
            "lambda": lam,
            "target_C": target_C,
            "subsets": [min_k_subset, large_k_subset],
        })

        logger.info(
            f"  min_k: k={min_k_subset['k']}, C={min_k_subset['C']:.4f} "
            f"(target={target_C:.4f}, delta={abs(min_k_subset['C'] - target_C):.4f})"
        )
        logger.info(
            f"  large_k: k={large_k_subset['k']}, C={large_k_subset['C']:.4f} "
            f"(target={target_C:.4f}, delta={abs(large_k_subset['C'] - target_C):.4f})"
        )

    result = {
        "benchmark": benchmark,
        "C_ref": C_ref,
        "levels": levels_output,
    }

    out_path = Path(output_dir) / "calibration" / benchmark / "capacity_matched_subsets.json"
    write_json(out_path, result)
    logger.info(f"Capacity-matched subsets saved -> {out_path}")

    # Save candidates
    candidate_rows = []
    for level in levels_output:
        for subset in level["subsets"]:
            candidate_rows.append({
                "lambda": level["lambda"],
                "target_C": level["target_C"],
                "type": subset["type"],
                "k": subset["k"],
                "C": subset["C"],
                "B_eff": subset["B_eff"],
                "N": subset["N"],
                "item_ids": "|".join(subset["item_ids"]),
            })
    pd.DataFrame(candidate_rows).to_csv(candidates_path, index=False)

    return result


def _find_min_k_subset(
    item_df: pd.DataFrame,
    correctness_tensor: dict,
    target_C: float,
    S_cal: float,
    tolerance: float,
    fallback_tolerance: float,
    epsilon: float,
) -> dict:
    """Greedy search: add items by descending discrimination until C ≈ target_C."""
    sorted_ids = item_df["item_id"].tolist()
    best = None
    best_delta = float("inf")

    for k in range(1, len(sorted_ids) + 1):
        subset_ids = sorted_ids[:k]
        r_pb_vals = item_df[item_df["item_id"].isin(subset_ids)]["r_pb"].tolist()
        B_eff = estimate_B_eff(r_pb_vals)
        N = estimate_N_subset(correctness_tensor, subset_ids, epsilon)
        C = compute_capacity(B_eff, S_cal, N)
        delta = abs(C - target_C)

        if delta < best_delta:
            best_delta = delta
            best = {
                "type": "min_k",
                "item_ids": subset_ids,
                "k": k,
                "B_eff": B_eff,
                "N": N,
                "S": S_cal,
                "C": C,
            }

        if delta <= tolerance * target_C:
            break

        # If C has overshot target significantly and started going away, break
        if k > 10 and C > target_C * (1 + fallback_tolerance) and delta > best_delta:
            break

    # Check if within tolerance
    if best and best_delta > tolerance * target_C:
        if best_delta <= fallback_tolerance * target_C:
            logger.warning(f"min_k: not within primary tolerance ({tolerance}), but within fallback ({fallback_tolerance})")
        else:
            logger.warning(f"min_k: delta={best_delta:.4f} exceeds fallback tolerance")

    return best or {"type": "min_k", "item_ids": [], "k": 0, "B_eff": 0, "N": epsilon, "S": S_cal, "C": 0}


def _find_large_k_subset(
    item_df: pd.DataFrame,
    correctness_tensor: dict,
    target_C: float,
    S_cal: float,
    min_k: int,
    tolerance: float,
    fallback_tolerance: float,
    epsilon: float,
) -> dict:
    """Find a larger subset (k > min_k) with C still close to target."""
    sorted_ids = item_df["item_id"].tolist()

    # Try increasing k values, starting from min_k+1
    best = None
    best_delta = float("inf")

    # Search among k values from min_k+1 to full benchmark
    step = max(1, (len(sorted_ids) - min_k) // 20)  # ~20 sample points

    for k in range(min_k + 1, len(sorted_ids) + 1, step):
        subset_ids = sorted_ids[:k]
        r_pb_vals = item_df[item_df["item_id"].isin(subset_ids)]["r_pb"].tolist()
        B_eff = estimate_B_eff(r_pb_vals)
        N = estimate_N_subset(correctness_tensor, subset_ids, epsilon)
        C = compute_capacity(B_eff, S_cal, N)
        delta = abs(C - target_C)

        if delta < best_delta:
            best_delta = delta
            best = {
                "type": "large_k",
                "item_ids": subset_ids,
                "k": k,
                "B_eff": B_eff,
                "N": N,
                "S": S_cal,
                "C": C,
            }

        if delta <= tolerance * target_C:
            break

    if best and best_delta > tolerance * target_C:
        if best_delta <= fallback_tolerance * target_C:
            logger.warning(f"large_k: tolerance relaxed to {fallback_tolerance}")
        else:
            logger.warning(f"large_k: delta={best_delta:.4f} exceeds fallback tolerance")

    return best or {"type": "large_k", "item_ids": [], "k": 0, "B_eff": 0, "N": epsilon, "S": S_cal, "C": 0}
