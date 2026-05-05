"""Compute calibration channel statistics for benchmarks."""
import argparse
import json
from pathlib import Path
from capacity_eval.src.config import load_experiment_config
from capacity_eval.src.calibration.item_statistics import compute_item_statistics
from capacity_eval.src.io_utils import write_json
from capacity_eval.src.logging_utils import setup_logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_config(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = _PROJECT_ROOT / "configs" / p.name
    return str(candidate) if candidate.exists() else str(p)


def main():
    parser = argparse.ArgumentParser(description="Compute calibration statistics")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark names")
    args = parser.parse_args()

    config = load_experiment_config(_resolve_config(args.config))
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    logger = setup_logger(log_file=f"{config.output_dir}/logs/run.log")
    logger.info(f"Computing calibration for: {benchmarks}")

    for benchmark in benchmarks:
        logger.info(f"Processing {benchmark}...")
        item_df, correctness_tensor = compute_item_statistics(
            output_dir=config.output_dir,
            benchmark=benchmark,
            epsilon=config.calibration.epsilon,
        )

        # Save correctness tensor for subset building
        tensor_path = Path(config.output_dir) / "calibration" / benchmark / "correctness_tensor.json"
        # Convert int keys to str for JSON serialization
        serializable = {}
        for model, repeats in correctness_tensor.items():
            serializable[model] = {str(k): v for k, v in repeats.items()}
        write_json(tensor_path, serializable)

    logger.info("Calibration computation complete")


if __name__ == "__main__":
    main()
