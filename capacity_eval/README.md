# Capacity Evaluation Framework

Benchmark evaluation capacity framework for NeurIPS evaluation theory paper.

## Setup

```bash
cd capacity_eval
pip install -r requirements.txt
```

Set API key environment variables before running:
```bash
source capacity_eval/set_keys.sh
```

or

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DASHSCOPE_API_KEY=sk-...
# etc.
```

## Full Pipeline

### 1. Run calibration repeated evaluation

LLM benchmarks:
```bash
python -m capacity_eval.src.cli.run_eval \
  --config configs/experiment.yaml \
  --model-config configs/models_llm.yaml \
  --group llm \
  --benchmarks mmlu_pro,agieval \
  --stage calibration \
  --repeats 1 \
  --max-workers 10
```

LMM benchmarks:
```bash
python -m capacity_eval.src.cli.run_eval \
  --config configs/experiment.yaml \
  --model-config configs/models_lmm.yaml \
  --group lmm \
  --benchmarks mmmu_pro,mmbench \
  --stage calibration \
  --repeats 5 \
  --max-workers 4
```

### 2. Compute calibration statistics

```bash
python -m capacity_eval.src.cli.compute_calibration \
  --config configs/experiment.yaml \
  --benchmarks mmlu_pro,agieval,mmmu_pro,mmbench
```

### 3. Build capacity-matched subsets (Exp 5.3)

```bash
python -m capacity_eval.src.cli.build_subsets \
  --config configs/experiment.yaml \
  --target-levels 0.1,0.2,0.3 \
  --tolerance 0.10
```

### 4. Run validation evaluation on held-out models

Full benchmark:
```bash
python -m capacity_eval.src.cli.run_eval \
  --config configs/experiment.yaml \
  --model-config configs/models_llm.yaml \
  --group llm \
  --benchmarks mmlu_pro,agieval \
  --stage validation \
  --repeats 1 \
  --max-workers 8
```

With subsets:
```bash
python -m capacity_eval.src.cli.run_eval \
  --config configs/experiment.yaml \
  --model-config configs/models_llm.yaml \
  --group llm \
  --benchmarks mmlu_pro,agieval \
  --stage validation \
  --subset-file outputs/calibration/{benchmark}/capacity_matched_subsets.json \
  --repeats 1 \
  --max-workers 8
```

### 5. Compute experiment metrics

```bash
# Exp 5.2
python -m capacity_eval.src.cli.compute_metrics \
  --config configs/experiment.yaml \
  --experiment exp52

# Exp 5.3
python -m capacity_eval.src.cli.compute_metrics \
  --config configs/experiment.yaml \
  --experiment exp53
```

## Output Structure

```
outputs/
  raw/           # Raw model responses (jsonl)
  parsed/        # Parsed answers with correctness (jsonl)
  scores/        # Aggregated scores and experiment results (csv)
  calibration/   # Item statistics, capacity, subset definitions
  logs/          # Run logs
```

## Key Concepts

- **C = B_eff * log2(1 + S/N)**: Evaluation capacity
- **B_eff**: Effective bandwidth (sum of item discrimination coefficients)
- **S**: Inter-model signal variance
- **N**: Evaluation noise (from repeated observations)
- **Calibration models (10)**: Used for estimating B_eff, S, N
- **Validation models (20)**: Held-out, used only for ranking fidelity assessment
