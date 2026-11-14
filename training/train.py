"""Standalone training run (same stages as the DAG). Example:
    python -m training.train --n-estimators 50 --max-depth 4 --export models/model.joblib
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from training import steps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/dataset.csv")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--f1-threshold", type=float, default=0.93)
    p.add_argument("--run-name", default=None)
    p.add_argument("--export", default=None, help="copy the trained model here (for Docker)")
    args = p.parse_args(argv)

    params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "seed": args.seed}
    with tempfile.TemporaryDirectory() as work:
        meta = steps.validate_dataset(args.dataset)
        prep = steps.preprocess(meta, work, args.seed)
        trained = steps.train_model(prep, params)
        result = steps.evaluate_and_log(trained, args.run_name)
        if args.export:
            Path(args.export).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(trained["model_path"], args.export)
    print(f"run_id={result['run_id']} f1={result['metrics']['f1']:.4f}")
    try:
        steps.quality_gate(result, args.f1_threshold)
        print("quality gate: passed")
    except ValueError as exc:
        print(f"quality gate: FAILED ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
