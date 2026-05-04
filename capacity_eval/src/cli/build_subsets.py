"""Build capacity-matched subsets for Experiment 5.3."""
import argparse
from capacity_eval.src.config import load_experiment_config
from capacity_eval.src.calibration.subset_builder import build_capacity_matched_subsets
from capacity_eval.src.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Build capacity-matched subsets")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--target-levels", default="0.1,0.2,0.3", help="Comma-separated target level lambdas")
    parser.add_argument("--tolerance", type=float, default=0.10, help="Primary tolerance for C matching")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmarks (default: all)")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    target_levels = [float(x) for x in args.target_levels.split(",")]
    all_benchmarks = config.benchmarks.llm + config.benchmarks.lmm
    benchmarks = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else all_benchmarks

    logger = setup_logger(log_file=f"{config.output_dir}/logs/run.log")
    logger.info(f"Building subsets for: {benchmarks}, levels: {target_levels}")

    for benchmark in benchmarks:
        result = build_capacity_matched_subsets(
            output_dir=config.output_dir,
            benchmark=benchmark,
            target_levels=target_levels,
            tolerance=config.exp53.tolerance,
            fallback_tolerance=config.exp53.fallback_tolerance,
            epsilon=config.calibration.epsilon,
        )

    logger.info("Subset building complete")


if __name__ == "__main__":
    main()
