from __future__ import annotations
import time
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from capacity_eval.src.config import ExperimentConfig, ModelConfig
from capacity_eval.src.datasets.base import BaseBenchmarkDataset, MCQItem
from capacity_eval.src.datasets import DATASET_REGISTRY
from capacity_eval.src.models.base import BaseModelClient
from capacity_eval.src.models.registry import create_model_client
from capacity_eval.src.prompting.templates import build_prompt
from capacity_eval.src.prompting.parser import parse_mcq_answer
from capacity_eval.src.io_utils import append_jsonl, get_completed_item_ids, ensure_dir
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def _make_record(
    run_id: str,
    benchmark: str,
    model_name: str,
    model_group: str,
    stage: str,
    repeat_id: int,
    item: MCQItem,
    prompt: str,
    raw_response: str,
    parsed_answer: str | None,
    correct: bool,
    latency: float,
    error: str | None,
) -> dict:
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "model": model_name,
        "model_group": model_group,
        "stage": stage,
        "repeat_id": repeat_id,
        "item_id": item.id,
        "question": item.question,
        "choices": item.choices,
        "answer": item.answer,
        "image_path": item.image_path,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "correct": correct,
        "latency": latency,
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
    }


class EvaluationRunner:
    def __init__(self, config: ExperimentConfig, model_configs: list[ModelConfig], group: str):
        self.config = config
        self.model_configs = model_configs
        self.group = group
        self.output_dir = Path(config.output_dir)

    def run(
        self,
        benchmarks: list[str],
        stage: str,
        repeats: int,
        max_workers: int = 4,
        subset_file: str | None = None,
        dry_run: bool = False,
    ) -> None:
        bm_config = self._load_benchmark_config()
        for bm_name in benchmarks:
            logger.info(f"=== Running benchmark: {bm_name}, stage: {stage}, repeats: {repeats} ===")
            dataset = self._load_dataset(bm_name, bm_config)

            items = dataset.get_all_items()
            if subset_file:
                items = self._filter_items_by_subset(items, bm_name, subset_file)

            for mcfg in self.model_configs:
                if stage == "calibration" and mcfg.group != "calibration":
                    continue
                if stage == "validation" and mcfg.group != "validation":
                    continue

                client = create_model_client(mcfg)
                has_image = dataset.get_item_by_id(items[0].id).image_path is not None if items else False

                for rep in range(repeats):
                    run_id = f"{bm_name}_{mcfg.name}_{stage}_r{rep}"
                    raw_path = self.output_dir / "raw" / bm_name / mcfg.name / f"{run_id}.jsonl"
                    parsed_path = self.output_dir / "parsed" / bm_name / mcfg.name / f"{run_id}.jsonl"

                    completed_ids = get_completed_item_ids(raw_path)
                    remaining = [it for it in items if it.id not in completed_ids]

                    logger.info(f"Model: {mcfg.name}, repeat: {rep}, items: {len(remaining)}/{len(items)} remaining")

                    if dry_run:
                        logger.info(f"[DRY RUN] Would evaluate {len(remaining)} items")
                        continue

                    self._evaluate_items(
                        client=client,
                        items=remaining,
                        benchmark=bm_name,
                        model_name=mcfg.name,
                        model_group=self.group,
                        stage=stage,
                        repeat_id=rep,
                        run_id=run_id,
                        raw_path=raw_path,
                        parsed_path=parsed_path,
                        has_image=has_image,
                        max_workers=max_workers,
                    )

    def _evaluate_items(
        self,
        client: BaseModelClient,
        items: list[MCQItem],
        benchmark: str,
        model_name: str,
        model_group: str,
        stage: str,
        repeat_id: int,
        run_id: str,
        raw_path: Path,
        parsed_path: Path,
        has_image: bool,
        max_workers: int,
    ) -> None:
        ensure_dir(raw_path.parent)
        ensure_dir(parsed_path.parent)

        def process_item(item: MCQItem) -> dict:
            prompt = build_prompt(item.question, item.choices, has_image=item.image_path is not None)
            raw_response = ""
            error_msg = None
            latency = 0.0
            try:
                t0 = time.time()
                raw_response = client.generate(prompt, image_path=item.image_path)
                latency = time.time() - t0
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{model_name}] Error on item {item.id}: {e}")

            valid_opts = set(item.choices.keys())
            parsed_answer = parse_mcq_answer(raw_response, valid_options=valid_opts) if raw_response else None
            correct = (parsed_answer == item.answer) if parsed_answer is not None else False

            record = _make_record(
                run_id=run_id,
                benchmark=benchmark,
                model_name=model_name,
                model_group=model_group,
                stage=stage,
                repeat_id=repeat_id,
                item=item,
                prompt=prompt,
                raw_response=raw_response,
                parsed_answer=parsed_answer,
                correct=correct,
                latency=latency,
                error=error_msg,
            )
            return record

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): item for item in items}
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(raw_path, record)
                if record.get("parsed_answer") is not None or record.get("error"):
                    append_jsonl(parsed_path, record)

    def _load_benchmark_config(self) -> dict:
        from capacity_eval.src.config import load_benchmarks_config
        bm_yaml = Path(__file__).parent.parent.parent / "configs" / "benchmarks.yaml"
        return load_benchmarks_config(bm_yaml) if bm_yaml.exists() else {}

    def _load_dataset(self, benchmark: str, bm_config: dict) -> BaseBenchmarkDataset:
        cls_name = bm_config.get(benchmark, {}).get("class", "")
        dataset_cls = DATASET_REGISTRY.get(benchmark)
        if dataset_cls is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        data_dir = getattr(self.config.data, benchmark, self.config.data.mmlu_pro)
        ds = dataset_cls(data_dir=data_dir)
        ds.load()
        return ds

    def _filter_items_by_subset(self, items: list[MCQItem], benchmark: str, subset_file: str) -> list[MCQItem]:
        from capacity_eval.src.io_utils import read_json
        data = read_json(subset_file)
        item_ids = set()
        for level in data.get("levels", []):
            for subset in level.get("subsets", []):
                item_ids.update(subset.get("item_ids", []))
        filtered = [it for it in items if it.id in item_ids]
        logger.info(f"Filtered to {len(filtered)} items from subset file")
        return filtered
