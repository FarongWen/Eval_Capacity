"""Compute experiment metrics (exp52 or exp53)."""
import argparse
from capacity_eval.src.config import load_experiment_config, load_model_configs
from capacity_eval.src.experiments.exp52_capacity_fidelity import run_capacity_fidelity_experiment
from capacity_eval.src.experiments.exp53_capacity_consistency import run_capacity_consistency_experiment
from capacity_eval.src.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Compute experiment metrics")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--model-config", default=None, help="Path to model config yaml (for exp53)")
    parser.add_argument("--experiment", required=True, choices=["exp52", "exp53"], help="Which experiment to compute")
    parser.add_argument("--group", default="llm", choices=["llm", "lmm"], help="Model group")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    logger = setup_logger(log_file=f"{config.output_dir}/logs/run.log")

    if args.experiment == "exp52":
        logger.info("Running Experiment 5.2: Capacity-Fidelity Relation")
        run_capacity_fidelity_experiment(config, group=args.group)
    elif args.experiment == "exp53":
        logger.info("Running Experiment 5.3: Capacity-Level Consistency")
        run_capacity_consistency_experiment(config, group=args.group)

    logger.info(f"{args.experiment} computation complete")


if __name__ == "__main__":
    main()
