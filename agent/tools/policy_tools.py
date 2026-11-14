"""Deterministic retraining-evidence check. It never triggers anything."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import get_settings
from agent.tools import airflow_tools
from agent.tools.base import tool
from agent.tools.mlflow_tools import _client, resolve_run

TRAINING_DAG = "ml_training_pipeline"


@tool(
    source="policy",
    description="Assess whether retraining is justified for a model run. Retraining is only "
                "recommended on evidence: freshness policy exceeded, the latest training run "
                "failed its quality gate, or the user supplied degradation results. Read-only.",
    properties={"run_id": {"type": "string"},
                "degradation_reported": {"type": "boolean"},
                "dag_id": {"type": "string"}},
    required=("run_id",),
)
def assess_retraining_need(run_id: str, degradation_reported: bool = False,
                           dag_id: str = TRAINING_DAG) -> dict[str, Any]:
    s = get_settings()
    run, _ = resolve_run(_client(), run_id)
    reasons: list[str] = []
    unknown: list[str] = []

    end = run.info.end_time
    age_days = None
    if end:
        age_days = (datetime.now(timezone.utc)
                    - datetime.fromtimestamp(end / 1000, tz=timezone.utc)).days
        if age_days > s.model_freshness_days:
            reasons.append(f"model is {age_days} days old; freshness policy is "
                           f"{s.model_freshness_days} days")
    else:
        unknown.append("run has no end time, so model age is unknown")

    try:
        latest = airflow_tools.latest_run_summary(dag_id)
        failed_gate = [t for t in latest["failed_tasks"] if t["task_id"] == "quality_gate"]
        if failed_gate:
            reasons.append("latest training run failed the quality gate")
        if latest["latest_run"] is None:
            unknown.append("no Airflow training runs found")
    except Exception as exc:  # noqa: BLE001
        unknown.append(f"Airflow status unavailable ({type(exc).__name__})")

    if degradation_reported:
        reasons.append("user supplied model-degradation results")
    else:
        unknown.append("no live-performance/degradation data was supplied")

    return {"retraining_recommended": bool(reasons), "reasons": reasons, "unknowns": unknown,
            "model_age_days": age_days,
            "policy": {"freshness_days": s.model_freshness_days}, "auto_triggered": False}
