from types import SimpleNamespace

from agent.tools import mlflow_tools, policy_tools
from agent.tools.base import execute_tool
from tests.conftest import make_run

PROV = {"dvc.md5": "bbbb", "git.commit": "a4c92e1", "dataset.version": "dataset-v1.1"}


class FakeClient:
    def __init__(self, runs):
        self.runs = runs

    def search_experiments(self, **kw):
        return [SimpleNamespace(experiment_id="1", name="demo")]

    def search_runs(self, experiment_ids, filter_string="", max_results=10, order_by=None, **kw):
        return self.runs[:max_results]

    def get_run(self, run_id):
        return next(r for r in self.runs if r.info.run_id == run_id)

    def search_model_versions(self, filter_string):
        return []


def test_best_run_ranks_by_metric_and_never_claims_production_readiness(monkeypatch):
    runs = [make_run("low", 0.90, tags=PROV), make_run("top", 0.97, tags=PROV),
            make_run("close", 0.968, tags={})]
    monkeypatch.setattr(mlflow_tools, "_client", lambda: FakeClient(runs))
    res = execute_tool("mlflow_best_run", {"metric": "f1"}, request_id="m1")
    assert res.data["best"]["run_id"] == "top"
    checks = {c["check"]: c["passed"] for c in res.data["readiness_checks"]}
    assert checks["registered in model registry"] is False
    assert checks["margin over runner-up > 0.005 (else likely noise)"] is False
    assert "does not make a model production-ready" in res.data["caveat"]


def test_best_run_accepts_common_f1_metric_alias(monkeypatch):
    runs = [make_run("low", 0.90, tags=PROV), make_run("top", 0.97, tags=PROV)]
    monkeypatch.setattr(mlflow_tools, "_client", lambda: FakeClient(runs))
    res = execute_tool("mlflow_best_run", {"metric": "f1_score"}, request_id="m4")
    assert res.status == "ok"
    assert res.data["metric"] == "f1"
    assert res.data["best"]["run_id"] == "top"


def test_dvc_provenance_is_read_from_the_mlflow_run(monkeypatch):
    monkeypatch.setattr(mlflow_tools, "_client", lambda: FakeClient([make_run("r25", 0.9, tags=PROV)]))
    from agent.tools import dvc_tools
    monkeypatch.setattr(dvc_tools, "_client", lambda: FakeClient([make_run("r25", 0.9, tags=PROV)]))
    res = execute_tool("dvc_get_run_dataset", {"reference": "r25"}, request_id="m2")
    assert res.data["dvc.md5"] == "bbbb" and res.data["dataset.version"] == "dataset-v1.1"


def test_missing_provenance_is_an_error_not_a_guess(monkeypatch):
    from agent.tools import dvc_tools
    monkeypatch.setattr(dvc_tools, "_client", lambda: FakeClient([make_run("r1", 0.9)]))
    res = execute_tool("dvc_get_run_dataset", {"reference": "r1"}, request_id="m3")
    assert res.status == "error" and res.error_code == "no_provenance"


def test_retraining_needs_evidence_and_is_never_auto_triggered(monkeypatch):
    fresh_ms = int(__import__("time").time() * 1000)
    monkeypatch.setattr(policy_tools, "_client", lambda: FakeClient([make_run("r1", 0.97, end_ms=fresh_ms)]))
    monkeypatch.setattr(policy_tools.airflow_tools, "latest_run_summary",
                        lambda dag: {"latest_run": {"state": "success"}, "failed_tasks": []})
    res = execute_tool("assess_retraining_need", {"run_id": "r1"}, request_id="p1")
    assert res.data["retraining_recommended"] is False and res.data["auto_triggered"] is False

    stale = make_run("r2", 0.97)  # ended in 2023 -> older than the 30-day freshness policy
    monkeypatch.setattr(policy_tools, "_client", lambda: FakeClient([stale]))
    res = execute_tool("assess_retraining_need", {"run_id": "r2"}, request_id="p2")
    assert res.data["retraining_recommended"] is True
    assert any("freshness policy" in r for r in res.data["reasons"])
