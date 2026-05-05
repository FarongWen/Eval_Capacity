"""Quick smoke test for all models in an LMM/LLM config."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from capacity_eval.src.config import load_model_configs
from capacity_eval.src.models.api_model import APIModelClient

PROMPT = "What is 1+1? Return only the number."


def test_config(config_path: str):
    models = load_model_configs(config_path)
    print(f"Testing {len(models)} models from {config_path}\n")

    results = []
    for mcfg in models:
        api_key = os.environ.get(mcfg.api_key_env, "")
        if not api_key:
            print(f"  [SKIP] {mcfg.name:40s}  key env {mcfg.api_key_env} not set")
            results.append((mcfg.name, "SKIP", "no key"))
            continue

        client = APIModelClient(mcfg, max_retries=2, retry_delay=1.0)
        try:
            t0 = time.time()
            resp = client.generate(PROMPT)
            latency = time.time() - t0
            ok = "2" in resp
            status = "OK" if ok else "WARN"
            note = f"resp={resp!r:20s} latency={latency:.1f}s"
        except Exception as e:
            status = "FAIL"
            note = str(e)[:80]
            latency = -1

        tag = {"OK": "✓", "WARN": "?", "FAIL": "✗", "SKIP": "-"}[status]
        print(f"  [{tag}] {mcfg.name:40s}  {note}")
        results.append((mcfg.name, status, note))

    ok = sum(1 for _, s, _ in results if s == "OK")
    warn = sum(1 for _, s, _ in results if s == "WARN")
    fail = sum(1 for _, s, _ in results if s == "FAIL")
    skip = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"\nSummary: {ok} OK, {warn} WARN, {fail} FAIL, {skip} SKIP / {len(results)} total")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True, help="Path to models yaml")
    args = parser.parse_args()
    test_config(args.model_config)
