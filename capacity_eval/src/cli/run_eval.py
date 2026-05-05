"""Run evaluation (calibration or validation) for specified benchmarks and models."""
import argparse
from pathlib import Path
from capacity_eval.src.config import load_experiment_config, load_model_configs
from capacity_eval.src.evaluation.runner import EvaluationRunner
from capacity_eval.src.logging_utils import setup_logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_config(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    # Try relative to project configs dir
    candidate = _PROJECT_ROOT / "configs" / p.name
    if candidate.exists():
        return str(candidate)
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--model-config", required=True, help="Path to models_llm.yaml or models_lmm.yaml")
    parser.add_argument("--group", required=True, choices=["llm", "lmm"], help="Model group")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark names")
    parser.add_argument("--stage", required=True, choices=["calibration", "validation"], help="Evaluation stage")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats")
    parser.add_argument("--max-workers", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("--subset-file", default=None, help="Path to capacity_matched_subsets.json (for validation with subsets)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    config = load_experiment_config(_resolve_config(args.config))
    model_configs = load_model_configs(_resolve_config(args.model_config))
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    logger = setup_logger(log_file=f"{config.output_dir}/logs/run.log")
    logger.info(f"Starting evaluation: group={args.group}, benchmarks={benchmarks}, stage={args.stage}, repeats={args.repeats}")

    runner = EvaluationRunner(config=config, model_configs=model_configs, group=args.group)
    runner.run(
        benchmarks=benchmarks,
        stage=args.stage,
        repeats=args.repeats,
        max_workers=args.max_workers,
        subset_file=args.subset_file,
        dry_run=args.dry_run,
    )
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
