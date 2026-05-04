from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.datasets.mmlu_pro import MMLUProDataset
from capacity_eval.src.datasets.agieval import AGIEvalDataset
from capacity_eval.src.datasets.mmmu_pro import MMMUProDataset
from capacity_eval.src.datasets.mmbench import MMBenchDataset

DATASET_REGISTRY: dict[str, type[BaseBenchmarkDataset]] = {
    "mmlu_pro": MMLUProDataset,
    "agieval": AGIEvalDataset,
    "mmmu_pro": MMMUProDataset,
    "mmbench": MMBenchDataset,
}

__all__ = ["BaseBenchmarkDataset", "MCQItem", "DATASET_REGISTRY", "MMLUProDataset", "AGIEvalDataset", "MMMUProDataset", "MMBenchDataset"]
