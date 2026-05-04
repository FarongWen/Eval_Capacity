from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    mmlu_pro: str = ""
    agieval: str = ""
    mmbench: str = ""
    mmmu_pro: str = ""


@dataclass
class CalibrationConfig:
    repeats: int = 5
    epsilon: float = 1e-8


@dataclass
class ValidationConfig:
    repeats: int = 1


@dataclass
class Exp53Config:
    target_levels: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    tolerance: float = 0.10
    fallback_tolerance: float = 0.15
    subset_types: list[str] = field(default_factory=lambda: ["min_k", "large_k"])


@dataclass
class BenchmarkGroupConfig:
    llm: list[str] = field(default_factory=lambda: ["mmlu_pro", "agieval"])
    lmm: list[str] = field(default_factory=lambda: ["mmmu_pro", "mmbench"])


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    random_seed: int = 42
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    exp53: Exp53Config = field(default_factory=Exp53Config)
    benchmarks: BenchmarkGroupConfig = field(default_factory=BenchmarkGroupConfig)


@dataclass
class ModelConfig:
    name: str = ""
    type: str = "api"
    provider: str = "openai"
    model_id: str = ""
    api_key_env: str = ""
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 10
    supports_vision: bool = False
    group: str = "calibration"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    data_cfg = DataConfig(**raw.get("data", {}))
    cal_cfg = CalibrationConfig(**raw.get("calibration", {}))
    val_cfg = ValidationConfig(**raw.get("validation", {}))
    exp53_cfg = Exp53Config(**raw.get("exp53", {}))
    bm_cfg = BenchmarkGroupConfig(**raw.get("benchmarks", {}))
    return ExperimentConfig(
        data=data_cfg,
        output_dir=raw.get("output_dir", "./outputs"),
        random_seed=raw.get("random_seed", 42),
        calibration=cal_cfg,
        validation=val_cfg,
        exp53=exp53_cfg,
        benchmarks=bm_cfg,
    )


def load_model_configs(path: str | Path) -> list[ModelConfig]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return [ModelConfig(**m) for m in raw.get("models", [])]


def load_benchmarks_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f).get("benchmarks", {})
