"""MLflow experiment-intelligence tools (MlflowClient only; never scrapes the UI)."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from agent.config import get_settings
from agent.tools.base import ToolError, tool

PROVENANCE_KEYS = ("dvc.path", "dvc.md5", "git.commit", "dataset.version")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\- ]+$")
_METRIC_ALIASES = {"f1_score": "f1", "f1-score": "f1", "f1score": "f1"}


def _client():
    # Fail fast instead of MLflow's default long retry/backoff when the server is down.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None


def _safe(value: str, what: str) -> str:
    if not _NAME_RE.match(value):
        raise ToolError("invalid_arguments", f"Invalid {what}: {value!r}")
    return value


def _summary(run) -> dict[str, Any]:
    info, data = run.info, run.data
    return {
        "run_id": info.run_id, "run_name": info.run_name, "experiment_id": info.experiment_id,
        "status": info.status, "start_time": _iso(info.start_time),
        "end_time": _iso(info.end_time), "metrics": dict(data.metrics),
        "params": dict(data.params),
        "provenance": {k: data.tags.get(k) for k in PROVENANCE_KEYS},
        "quality_gate": data.tags.get("quality_gate"),
    }


def _experiment_ids(client, name: str | None) -> list[str]:
    if name:
        exp = client.get_experiment_by_name(_safe(name, "experiment name"))
        if exp is None:
            raise ToolError("not_found", f"Experiment '{name}' does not exist")
        return [exp.experiment_id]
    return [e.experiment_id for e in client.search_experiments()]


def resolve_run(client, reference: str):
    """Resolve a run id, a numeric experiment id (latest run) or a run name."""
    _safe(reference, "run reference")
    try:
        return client.get_run(reference), "run_id"
    except Exception:  # noqa: BLE001 - fall through to other interpretations
        pass
    if reference.isdigit():
        runs = client.search_runs([reference], max_results=1,
                                  order_by=["attributes.start_time DESC"])
        if runs:
            return runs[0], "latest_run_of_experiment_id"
    runs = client.search_runs(_experiment_ids(client, None), max_results=1,
                              filter_string=f"attributes.run_name = '{reference}'",
                              order_by=["attributes.start_time DESC"])
    if runs:
        return runs[0], "run_name"
    raise ToolError("not_found", f"No run, experiment id or run name matches '{reference}'")


@tool(source="mlflow", description="List MLflow experiments.")
def mlflow_list_experiments() -> dict[str, Any]:
    exps = _client().search_experiments()
    return {"experiments": [{"experiment_id": e.experiment_id, "name": e.name} for e in exps]}


@tool(
    source="mlflow",
    description="List the most recent training runs (newest first) with metrics and params.",
    properties={"experiment_name": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50}},
)
def mlflow_list_runs(experiment_name: str | None = None, max_results: int = 10) -> dict[str, Any]:
    client = _client()
    runs = client.search_runs(_experiment_ids(client, experiment_name), max_results=max_results,
                              order_by=["attributes.start_time DESC"])
    return {"runs": [_summary(r) for r in runs], "count": len(runs)}


@tool(
    source="mlflow",
    description="Get one run's parameters, metrics, provenance tags and artifact list.",
    properties={"run_id": {"type": "string"}},
    required=("run_id",),
)
def mlflow_get_run(run_id: str) -> dict[str, Any]:
    client = _client()
    run, _ = resolve_run(client, run_id)
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    return {**_summary(run), "artifacts": artifacts}


def _readiness(client, run, metric: str, ranked: list) -> list[dict[str, Any]]:
    s = get_settings()
    checks = [{"check": "run finished", "passed": run.info.status == "FINISHED"}]
    checks.append({"check": "dataset provenance recorded",
                   "passed": bool(run.data.tags.get("dvc.md5") and run.data.tags.get("git.commit"))})
    if metric == "f1":
        checks.append({"check": f"f1 >= quality gate ({s.quality_gate_f1})",
                       "passed": run.data.metrics.get("f1", 0) >= s.quality_gate_f1})
    try:
        versions = client.search_model_versions(f"run_id = '{run.info.run_id}'")
        checks.append({"check": "registered in model registry", "passed": bool(versions)})
    except Exception:  # noqa: BLE001
        checks.append({"check": "registered in model registry", "passed": None})
    if len(ranked) > 1:
        gap = abs(ranked[0].data.metrics[metric] - ranked[1].data.metrics[metric])
        checks.append({"check": "margin over runner-up > 0.005 (else likely noise)",
                       "passed": gap > 0.005, "gap": round(gap, 5)})
    return checks


@tool(
    source="mlflow",
    description="Find the best run among the most recent finished runs for a metric. Returns the "
                "ranking plus readiness checks. A top metric alone does NOT make a model "
                "production-ready; report the checks and any failures. This project's training "
                "pipeline logs F1 as 'f1' (not 'f1_score'); it also logs accuracy, precision, "
                "and recall.",
    properties={"metric": {"type": "string",
                           "description": "Metric name as logged; use 'f1' for F1 in this project."},
                "mode": {"type": "string", "enum": ["max", "min"]},
                "experiment_name": {"type": "string"},
                "recent_runs": {"type": "integer", "minimum": 2, "maximum": 50}},
)
def mlflow_best_run(metric: str = "f1", mode: str = "max", experiment_name: str | None = None,
                    recent_runs: int = 10) -> dict[str, Any]:
    metric = _METRIC_ALIASES.get(metric.strip().casefold(), metric)
    client = _client()
    runs = client.search_runs(_experiment_ids(client, experiment_name),
                              filter_string="attributes.status = 'FINISHED'",
                              max_results=recent_runs, order_by=["attributes.start_time DESC"])
    candidates = [r for r in runs if metric in r.data.metrics]
    if not candidates:
        raise ToolError("not_found", f"No recent finished runs logged metric '{metric}'")
    ranked = sorted(candidates, key=lambda r: r.data.metrics[metric], reverse=(mode == "max"))
    return {
        "metric": metric, "mode": mode, "runs_considered": len(candidates),
        "best": _summary(ranked[0]),
        "ranking": [{"run_id": r.info.run_id, "run_name": r.info.run_name,
                     metric: r.data.metrics[metric]} for r in ranked[:5]],
        "readiness_checks": _readiness(client, ranked[0], metric, ranked),
        "caveat": "The highest metric does not make a model production-ready. "
                  "Review readiness_checks before promotion.",
    }


@tool(
    source="mlflow",
    description="Compare specific runs side by side (metrics and params), with deltas vs the first.",
    properties={"run_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                            "maxItems": 10}},
    required=("run_ids",),
)
def mlflow_compare_runs(run_ids: list[str]) -> dict[str, Any]:
    client = _client()
    runs = [resolve_run(client, r)[0] for r in run_ids]
    base = runs[0].data.metrics
    rows = []
    for run in runs:
        row = _summary(run)
        row["metric_delta_vs_first"] = {k: round(v - base[k], 6) for k, v in
                                        run.data.metrics.items() if k in base}
        rows.append(row)
    return {"runs": rows}


@tool(
    source="mlflow",
    description="Compare experiments by their runs' best and mean value of a metric.",
    properties={"experiment_names": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                                     "maxItems": 5},
                "metric": {"type": "string"}},
    required=("experiment_names",),
)
def mlflow_compare_experiments(experiment_names: list[str], metric: str = "f1") -> dict[str, Any]:
    client = _client()
    out = []
    for name in experiment_names:
        runs = client.search_runs(_experiment_ids(client, name), max_results=100)
        values = [r.data.metrics[metric] for r in runs if metric in r.data.metrics]
        out.append({"experiment": name, "runs_with_metric": len(values),
                    "best": max(values) if values else None,
                    "mean": round(mean(values), 6) if values else None})
    return {"metric": metric, "experiments": out}


@tool(
    source="mlflow",
    description="Model registry info: registered models, versions, aliases and source runs.",
    properties={"model_name": {"type": "string"}},
)
def mlflow_model_registry(model_name: str | None = None) -> dict[str, Any]:
    client = _client()
    names = [_safe(model_name, "model name")] if model_name else [
        m.name for m in client.search_registered_models()]
    models = []
    for name in names:
        versions = client.search_model_versions(f"name = '{name}'")
        models.append({"name": name, "versions": [
            {"version": v.version, "run_id": v.run_id, "status": v.status,
             "aliases": list(getattr(v, "aliases", []) or []),
             "created": _iso(v.creation_timestamp)} for v in versions]})
    return {"models": models}
