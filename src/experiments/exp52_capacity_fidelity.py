from __future__ import annotations
import json
from pathlib import Path
from capacity_eval.src.config import ExperimentConfig, load_experiment_config, load_model_configs
from capacity_eval.src.evaluation.runner import EvaluationRunner
from capacity_eval.src.evaluation.scorer import compute_scores
from capacity_eval.src.evaluation.metrics import compute_ranking_fidelity
from capacity_eval.src.calibration.item_statistics import compute_item_statistics
from capacity_eval.src.calibration.subset_builder import build_capacity_matched_subsets
from capacity_eval.src.io_utils import read_json, write_json, ensure_dir
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def run_capacity_fidelity_experiment(config: ExperimentConfig, group: str) -> None:
    """Experiment 5.2: Capacity-Fidelity Relation.

    For each benchmark in the group, compute C and ranking fidelity at various settings.
    The settings (G0-G6) are configurable via yaml, defining different benchmark configurations.
    """
    benchmarks = config.benchmarks.llm if group == "llm" else config.benchmarks.lmm
    output_dir = Path(config.output_dir)

    results = []

    for benchmark in benchmarks:
        cal_dir = output_dir / "calibration" / benchmark
        cap_path = cal_dir / "capacity_full.json"

        if not cap_path.exists():
            logger.warning(f"Capacity data not found for {benchmark}, run compute_calibration first")
            continue

        cap_data = read_json(cap_path)
        C_ref = cap_data["C_ref"]
        B_eff = cap_data["B_eff"]
        S_cal = cap_data["S_cal"]
        N = cap_data["N"]

        ranking_df = compute_ranking_fidelity(str(output_dir), benchmark)

        if not ranking_df.empty:
            for _, row in ranking_df.iterrows():
                results.append({
                    "benchmark": benchmark,
                    "B_eff": B_eff,
                    "S_cal": S_cal,
                    "N": N,
                    "C": C_ref,
                    **row.to_dict(),
                })

    result_df_path = output_dir / "scores" / "exp52_capacity_fidelity.csv"
    ensure_dir(result_df_path.parent)
    import pandas as pd
    pd.DataFrame(results).to_csv(result_df_path, index=False)
    logger.info(f"Exp 5.2 results saved -> {result_df_path}")
