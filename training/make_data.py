"""Generate dataset versions for the DVC demo.

v1: breast-cancer dataset (569 rows). v2: v1 plus 300 jittered rows, so the DVC hash,
row count and size all change between versions.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build(version: str):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    frame = load_breast_cancer(as_frame=True).frame
    if version == "v2":
        rng = np.random.default_rng(7)
        extra = frame.sample(300, random_state=7, replace=True).copy()
        features = [c for c in extra.columns if c != "target"]
        extra[features] = extra[features] * rng.normal(1.0, 0.02, size=(300, len(features)))
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--out", default="data/dataset.csv")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build(args.version)
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows, version {args.version})")


if __name__ == "__main__":
    main()
