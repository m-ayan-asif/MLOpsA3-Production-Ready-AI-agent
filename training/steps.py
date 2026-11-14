"""Pipeline stages shared by the CLI (training/train.py) and the Airflow DAG."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

MODEL_NAME = "breast-cancer-rf"
MIN_ROWS = 100


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - DVC's content hash, not a security control
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dvc_record(dataset_path: str | Path) -> dict[str, Any]:
    dvc_file = Path(str(dataset_path) + ".dvc")
    if not dvc_file.is_file():
        return {}
    doc = yaml.safe_load(dvc_file.read_text()) or {}
    return (doc.get("outs") or [{}])[0]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=15,
                             check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def read_provenance(dataset_path: str | Path) -> dict[str, str]:
    """DVC hash + Git commit/tag recorded at training time (what the agent reads back)."""
    dvc_file = str(dataset_path) + ".dvc"
    commit = _git("log", "-1", "--format=%H", "--", dvc_file) or os.getenv("GIT_COMMIT", "unknown")
    tags = _git("tag", "--points-at", commit).split() if commit != "unknown" else []
    version = next((t for t in tags if t.startswith("dataset-")), tags[0] if tags else "untagged")
    return {"dvc.path": str(dataset_path), "dvc.md5": dvc_record(dataset_path).get("md5", "unknown"),
            "git.commit": commit, "dataset.version": version}


def validate_dataset(dataset_path: str) -> dict[str, Any]:
    import pandas as pd

    path = Path(dataset_path)
    if not path.is_file():
        raise ValueError(f"dataset not found: {path}")
    recorded = dvc_record(path).get("md5")
    actual = file_md5(path)
    if recorded and recorded != actual:
        raise ValueError(f"dataset hash {actual} does not match DVC-recorded {recorded}; "
                         "tracked data has been modified (run `dvc checkout`)")
    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError("missing 'target' column")
    if len(frame) < MIN_ROWS:
        raise ValueError(f"only {len(frame)} rows (< {MIN_ROWS})")
    if frame.isnull().any().any():
        raise ValueError("dataset contains null values")
    return {"dataset_path": str(path), "md5": actual, "rows": len(frame)}


def preprocess(meta: dict[str, Any], work_dir: str, seed: int = 42) -> dict[str, Any]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(meta["dataset_path"])
    train, test = train_test_split(frame, test_size=0.2, random_state=seed,
                                   stratify=frame["target"])
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(work_dir) / "train.csv", index=False)
    test.to_csv(Path(work_dir) / "test.csv", index=False)
    return {**meta, "work_dir": work_dir, "train_rows": len(train), "test_rows": len(test)}


def train_model(prep: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    train = pd.read_csv(Path(prep["work_dir"]) / "train.csv")
    model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                   max_depth=params["max_depth"], random_state=params["seed"])
    model.fit(train.drop(columns="target"), train["target"])
    model_path = str(Path(prep["work_dir"]) / "model.joblib")
    joblib.dump(model, model_path)
    return {**prep, "model_path": model_path, "params": params}


def evaluate_and_log(trained: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    import joblib
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    test = pd.read_csv(Path(trained["work_dir"]) / "test.csv")
    model = joblib.load(trained["model_path"])
    pred = model.predict(test.drop(columns="target"))
    y = test["target"]
    metrics = {"f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred),
               "precision": precision_score(y, pred), "recall": recall_score(y, pred)}

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mlops-agent-demo"))
    prov = read_provenance(trained["dataset_path"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**prov, "pipeline": os.getenv("PIPELINE_NAME", "manual")})
        mlflow.log_params(trained["params"])
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
    return {"run_id": run.info.run_id, "metrics": metrics,
            "model_version": getattr(info, "registered_model_version", None),
            "model_path": trained["model_path"]}


def quality_gate(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Pass -> tag the run and give the registered version the 'candidate' alias.
    Fail -> tag the run and raise, so nothing downstream is promoted.
    Promotion to production is deliberately NOT done here: that stays a human decision."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    f1 = result["metrics"]["f1"]
    if f1 < threshold:
        client.set_tag(result["run_id"], "quality_gate", "failed")
        raise ValueError(f"quality gate failed: f1={f1:.4f} < threshold {threshold}")
    client.set_tag(result["run_id"], "quality_gate", "passed")
    if result.get("model_version"):
        client.set_registered_model_alias(MODEL_NAME, "candidate", str(result["model_version"]))
    return {**result, "gate": "passed", "threshold": threshold}
