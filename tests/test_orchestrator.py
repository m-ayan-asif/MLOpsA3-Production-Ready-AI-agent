"""Tool selection, multi-tool reasoning, grounding when tools fail, confirmation gating."""
from agent.llm import ToolCall
from agent.orchestrator import Orchestrator
from tests.conftest import FakeLLM, calls


def test_selects_and_executes_the_requested_tool(stub_tool):
    stub_tool("mlflow_best_run", lambda **kw: {"best": {"run_id": "r1"}})
    llm = FakeLLM([calls("mlflow_best_run"), "Best run is r1."])
    result = Orchestrator(llm).run("which run is best?")
    assert [t.tool for t in result.tool_trace] == ["mlflow_best_run"]
    assert result.tool_trace[0].status == "ok"
    assert result.answer == "Best run is r1."
    assert result.tool_failures == []


def test_mandatory_scenario_uses_all_four_systems_without_triggering_retraining(
        stub_tool, monkeypatch):
    from agent.tools import airflow_tools

    stub_tool("mlflow_best_run", lambda **kw: {"best": {"run_id": "r1"}})
    stub_tool("dvc_get_run_dataset", lambda **kw: {"dataset.version": "dataset-v1.1"})
    stub_tool("k8s_get_deployed_version", lambda **kw: {"deployments": [{"ready": 2}]})
    stub_tool("airflow_latest_run_summary", lambda **kw: {"latest_run": {"state": "success"}})
    stub_tool("assess_retraining_need", lambda **kw: {"retraining_recommended": False})
    api_calls = []
    monkeypatch.setattr(airflow_tools, "_api", lambda *a, **kw: api_calls.append(a))

    llm = FakeLLM([
        calls("mlflow_best_run"),
        [ToolCall("dvc_get_run_dataset", {"reference": "r1"}),
         ToolCall("k8s_get_deployed_version", {}),
         ToolCall("airflow_latest_run_summary", {"dag_id": "ml_training_pipeline"}),
         ToolCall("assess_retraining_need", {"run_id": "r1"}),
         ToolCall("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"})],  # model over-reaches
        "Facts / Uncertainties / Actions",
    ])
    result = Orchestrator(llm).run("Find the best recent model, identify its dataset, check "
                                   "deployment, and say whether retraining is necessary.")
    assert {t.source for t in result.tool_trace} >= {"mlflow", "dvc", "kubernetes", "airflow"}
    trigger = [t for t in result.tool_trace if t.tool == "airflow_trigger_dag"][0]
    assert trigger.status == "confirmation_required"
    assert api_calls == []  # no POST to Airflow: retraining was never started


def test_agent_does_not_invent_results_when_a_tool_is_down(stub_tool):
    def down(**kw):
        raise ConnectionError("connection refused")

    stub_tool("mlflow_best_run", down)
    llm = FakeLLM([calls("mlflow_best_run"), "The best run is r42 with f1 0.99."])  # hallucination
    result = Orchestrator(llm).run("best model?")
    payload = llm.seen_tool_payloads[0][0][1]
    assert payload["status"] == "error" and payload["data"] is None  # LLM got no data
    assert result.tool_failures and "mlflow_best_run" in result.tool_failures[0]
    assert result.answer.startswith("\u26a0\ufe0f")
    assert "NOT verified" in result.answer


def test_answer_without_any_tool_call_is_flagged_as_unbacked():
    result = Orchestrator(FakeLLM(["Everything is fine."])).run("is prod healthy?")
    assert "No tools were called" in result.answer


def test_confirm_flag_is_controlled_by_the_request_not_the_llm(stub_tool):
    seen = []
    stub_tool("airflow_trigger_dag", lambda **kw: seen.append(kw) or {"triggered": True})

    def llm_that_claims_confirmation():
        call = ToolCall("airflow_trigger_dag", {"dag_id": "ml_training_pipeline", "confirm": True})
        return FakeLLM([[call], "done"])

    Orchestrator(llm_that_claims_confirmation()).run("retrain please")
    assert seen[0]["confirm"] is False  # the LLM's own confirm=True was overwritten
    seen.clear()
    Orchestrator(llm_that_claims_confirmation()).run("retrain please", confirm=True)
    assert seen[0]["confirm"] is True  # only the request-level flag enables it
