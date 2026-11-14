import requests

from agent.tools import airflow_tools
from agent.tools.base import execute_tool
from tests.conftest import FakeResponse


def _install(monkeypatch, routes, posts=None):
    posts = [] if posts is None else posts

    def fake_post(url, **kw):
        assert url.endswith("/auth/token")
        return FakeResponse({"access_token": "jwt-token"}, 201)

    def fake_request(method, url, **kw):
        assert kw["headers"]["Authorization"] == "Bearer jwt-token"
        if method == "POST":
            posts.append((url, kw.get("json")))
            return FakeResponse({"dag_run_id": "manual__1", "state": "queued"})
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return FakeResponse(payload)
        return FakeResponse({"detail": "nope"}, 404)

    monkeypatch.setattr(airflow_tools.requests, "post", fake_post)
    monkeypatch.setattr(airflow_tools.requests, "request", fake_request)
    return posts


def test_failed_tasks_are_identified_from_the_authenticated_api(monkeypatch):
    _install(monkeypatch, {"/taskInstances": {"task_instances": [
        {"task_id": "validate_dataset", "state": "success", "try_number": 1},
        {"task_id": "preprocess", "state": "failed", "try_number": 3},
        {"task_id": "train_model", "state": "upstream_failed", "try_number": 0}]}})
    res = execute_tool("airflow_get_failed_tasks",
                       {"dag_id": "ml_training_pipeline", "run_id": "manual__2026-11-14T10:00:00+00:00"},
                       request_id="a1")
    assert res.status == "ok"
    assert [t["task_id"] for t in res.data["failed"]] == ["preprocess", "train_model"]


def test_trigger_requires_confirmation_and_an_approved_dag(monkeypatch):
    posts = _install(monkeypatch, {})
    unconfirmed = execute_tool("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"},
                               request_id="a2")
    other = execute_tool("airflow_trigger_dag", {"dag_id": "some_other_dag"}, request_id="a3",
                         confirmed=True)
    bad_conf = execute_tool("airflow_trigger_dag", {"dag_id": "ml_training_pipeline",
                            "conf": {"shell": "rm -rf /"}}, request_id="a4", confirmed=True)
    assert unconfirmed.status == "confirmation_required"
    assert other.status == "denied" and bad_conf.status == "denied"
    assert posts == []


def test_confirmed_trigger_posts_to_the_dag_runs_endpoint(monkeypatch):
    posts = _install(monkeypatch, {})
    res = execute_tool("airflow_trigger_dag", {"dag_id": "ml_training_pipeline",
                       "conf": {"n_estimators": 50}}, request_id="a5", confirmed=True)
    assert res.status == "ok" and res.data["triggered"] is True
    assert posts[0][0].endswith("/api/v2/dags/ml_training_pipeline/dagRuns")
    assert posts[0][1]["conf"] == {"n_estimators": 50}


def test_unreachable_airflow_is_a_structured_error_without_leaking_credentials(monkeypatch):
    def refuse(*a, **kw):
        raise requests.ConnectionError("connection refused for super-secret-pw")

    monkeypatch.setattr(airflow_tools.requests, "post", refuse)
    res = execute_tool("airflow_list_dags", {}, request_id="a6")
    assert res.status == "error" and res.error_code == "service_unavailable"
    assert "super-secret-pw" not in (res.error or "")
