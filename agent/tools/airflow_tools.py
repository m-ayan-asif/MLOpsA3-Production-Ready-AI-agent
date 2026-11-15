"""Airflow 3 tools using the authenticated REST API (JWT from /auth/token, API at /api/v2)."""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import requests
from requests import RequestException

from agent.config import get_settings
from agent.tools.base import ToolError, tool

_token_cache: dict[str, Any] = {"token": None, "expires": 0.0}
FAILED_STATES = {"failed", "upstream_failed"}
ALLOWED_CONF_KEYS = {"n_estimators", "max_depth", "f1_threshold", "seed"}


def _token(force: bool = False) -> str:
    if not force and _token_cache["token"] and time.time() < _token_cache["expires"]:
        return _token_cache["token"]
    s = get_settings()
    try:
        resp = requests.post(f"{s.airflow_base_url}/auth/token", timeout=s.http_timeout,
                             json={"username": s.airflow_username, "password": s.airflow_password})
    except RequestException as exc:
        raise ToolError("service_unavailable",
                        f"Airflow unreachable at {s.airflow_base_url} ({type(exc).__name__})") from exc
    if resp.status_code not in (200, 201):
        raise ToolError("auth_failed", f"Airflow authentication failed (HTTP {resp.status_code})")
    _token_cache.update(token=resp.json()["access_token"], expires=time.time() + 600)
    return _token_cache["token"]


def _api(method: str, path: str, *, params: dict | None = None, body: dict | None = None):
    s = get_settings()
    for attempt in (0, 1):
        headers = {"Authorization": f"Bearer {_token(force=attempt == 1)}",
                   "Accept": "application/json"}
        try:
            resp = requests.request(method, f"{s.airflow_base_url}/api/v2{path}", headers=headers,
                                    params=params, json=body, timeout=s.http_timeout)
        except RequestException as exc:
            raise ToolError("service_unavailable",
                            f"Airflow unreachable ({type(exc).__name__})") from exc
        if resp.status_code == 401 and attempt == 0:
            continue  # token expired: refresh once
        break
    if resp.status_code == 404:
        raise ToolError("not_found", f"Airflow resource not found: {path}")
    if resp.status_code >= 400:
        raise ToolError("api_error", f"Airflow API HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _q(value: str) -> str:
    return quote(value, safe="")


def _run_view(r: dict[str, Any]) -> dict[str, Any]:
    keys = ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date", "conf")
    return {k: r.get(k) for k in keys}


def latest_run_summary(dag_id: str) -> dict[str, Any]:
    runs = _api("GET", f"/dags/{_q(dag_id)}/dagRuns", params={"limit": 20,
                                                              "order_by": "-start_date"})
    items = sorted(runs.get("dag_runs", []),
                   key=lambda r: r.get("start_date") or r.get("run_after") or "", reverse=True)
    if not items:
        return {"dag_id": dag_id, "latest_run": None, "failed_tasks": []}
    latest = items[0]
    tis = _api("GET", f"/dags/{_q(dag_id)}/dagRuns/{_q(latest['dag_run_id'])}/taskInstances")
    failed = [{"task_id": t["task_id"], "state": t["state"], "try_number": t.get("try_number")}
              for t in tis.get("task_instances", []) if t.get("state") in FAILED_STATES]
    return {"dag_id": dag_id, "latest_run": _run_view(latest), "failed_tasks": failed}


@tool(source="airflow", description="List Airflow DAG definitions and whether each is approved for "
                                   "triggering. This does not report any DAG run's status.")
def airflow_list_dags() -> dict[str, Any]:
    approved = set(get_settings().approved_dag_ids)
    dags = _api("GET", "/dags", params={"limit": 100}).get("dags", [])
    return {"dags": [{"dag_id": d["dag_id"], "is_paused": d.get("is_paused"),
                      "approved_for_trigger": d["dag_id"] in approved} for d in dags]}


@tool(
    source="airflow",
    description="Get recent run status for a DAG (newest first).",
    properties={"dag_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
    required=("dag_id",),
)
def airflow_get_dag_runs(dag_id: str, limit: int = 5) -> dict[str, Any]:
    data = _api("GET", f"/dags/{_q(dag_id)}/dagRuns", params={"limit": 50,
                                                              "order_by": "-start_date"})
    runs = sorted(data.get("dag_runs", []),
                  key=lambda r: r.get("start_date") or r.get("run_after") or "", reverse=True)
    return {"dag_id": dag_id, "runs": [_run_view(r) for r in runs[:limit]]}


@tool(
    source="airflow",
    description="Get the actual state of the latest run of a DAG and list its failed tasks. Use this "
                "for questions about whether a recent training workflow succeeded or failed.",
    properties={"dag_id": {"type": "string"}},
    required=("dag_id",),
)
def airflow_latest_run_summary(dag_id: str) -> dict[str, Any]:
    return latest_run_summary(dag_id)


@tool(
    source="airflow",
    description="List task states for one DAG run and identify failed tasks.",
    properties={"dag_id": {"type": "string"}, "run_id": {"type": "string"}},
    required=("dag_id", "run_id"),
)
def airflow_get_failed_tasks(dag_id: str, run_id: str) -> dict[str, Any]:
    tis = _api("GET", f"/dags/{_q(dag_id)}/dagRuns/{_q(run_id)}/taskInstances")
    tasks = [{"task_id": t["task_id"], "state": t["state"], "try_number": t.get("try_number"),
              "duration": t.get("duration")} for t in tis.get("task_instances", [])]
    return {"run_id": run_id, "tasks": tasks,
            "failed": [t for t in tasks if t["state"] in FAILED_STATES]}


@tool(
    source="airflow",
    description="Get execution info and the log tail for one task instance (failure diagnostics).",
    properties={"dag_id": {"type": "string"}, "run_id": {"type": "string"},
                "task_id": {"type": "string"}, "try_number": {"type": "integer", "minimum": 1},
                "tail_lines": {"type": "integer", "minimum": 5, "maximum": 200}},
    required=("dag_id", "run_id", "task_id"),
)
def airflow_get_task_info(dag_id: str, run_id: str, task_id: str, try_number: int | None = None,
                          tail_lines: int = 40) -> dict[str, Any]:
    base = f"/dags/{_q(dag_id)}/dagRuns/{_q(run_id)}/taskInstances/{_q(task_id)}"
    ti = _api("GET", base)
    attempt = try_number or ti.get("try_number") or 1
    raw = _api("GET", f"{base}/logs/{attempt}").get("content", [])
    lines = [f"{i.get('level', '')} {i.get('event', '')}".strip() if isinstance(i, dict)
             else str(i) for i in (raw if isinstance(raw, list) else str(raw).splitlines())]
    return {"task_id": task_id, "state": ti.get("state"), "try_number": attempt,
            "start_date": ti.get("start_date"), "end_date": ti.get("end_date"),
            "duration": ti.get("duration"), "log_tail": lines[-tail_lines:]}


@tool(
    source="airflow",
    mutating=True,
    description="Trigger an APPROVED training DAG run. Optional conf may only contain "
                "n_estimators, max_depth, f1_threshold, seed.",
    properties={"dag_id": {"type": "string"}, "conf": {"type": "object"}},
    required=("dag_id",),
)
def airflow_trigger_dag(dag_id: str, conf: dict[str, Any] | None = None,
                        confirm: bool = False) -> dict[str, Any]:
    if dag_id not in get_settings().approved_dag_ids:
        raise ToolError("denied", f"DAG '{dag_id}' is not approved for triggering")
    unknown = set(conf or {}) - ALLOWED_CONF_KEYS
    if unknown:
        raise ToolError("denied", f"conf keys not permitted: {sorted(unknown)}")
    if not confirm:
        raise ToolError("confirmation_required",
                        f"Triggering '{dag_id}' starts a real training run. Ask the user to confirm.")
    run = _api("POST", f"/dags/{_q(dag_id)}/dagRuns", body={"logical_date": None,
                                                           "conf": conf or {}})
    return {"triggered": True, **_run_view(run)}
