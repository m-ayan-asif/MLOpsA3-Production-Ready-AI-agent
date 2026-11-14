"""Timeouts, invalid requests, service failures and structured, secret-free logging."""
import logging
import time

from agent.tools.base import REGISTRY, execute_tool, tool


def test_tool_timeout_returns_structured_timeout(stub_tool):
    stub_tool("mlflow_list_experiments", lambda: time.sleep(0.6))
    res = execute_tool("mlflow_list_experiments", {}, request_id="t1", timeout=0.1)
    assert res.status == "timeout" and res.error_code == "timeout" and res.data is None


def test_invalid_arguments_are_rejected_before_execution(stub_tool):
    called = []
    stub_tool("mlflow_list_runs", lambda **kw: called.append(kw))
    res = execute_tool("mlflow_list_runs", {"max_results": 9999}, request_id="t2")
    assert res.status == "invalid_arguments" and called == []
    res = execute_tool("mlflow_list_runs", {"bogus": 1}, request_id="t2")
    assert res.status == "invalid_arguments"


def test_unknown_tool(request):
    assert execute_tool("drop_database", {}, request_id="t3").status == "unknown_tool"


def test_service_failure_is_reported_not_raised(stub_tool):
    def boom():
        raise ConnectionError("connection refused")

    stub_tool("mlflow_list_experiments", boom)
    res = execute_tool("mlflow_list_experiments", {}, request_id="t4")
    assert res.status == "error" and "ConnectionError" in res.error


def test_float_integers_from_llm_are_coerced(stub_tool):
    got = {}
    stub_tool("mlflow_list_runs", lambda **kw: got.update(kw) or {})
    execute_tool("mlflow_list_runs", {"max_results": 5.0}, request_id="t5")
    assert got["max_results"] == 5 and isinstance(got["max_results"], int)


def test_structured_logs_have_required_fields_and_no_secrets(monkeypatch, caplog):
    @tool(source="test", description="x",
          properties={"password": {"type": "string"}, "note": {"type": "string"}})
    def logging_probe(password: str = "", note: str = ""):
        return {"ok": True}

    try:
        with caplog.at_level(logging.INFO, logger="mlops_agent"):
            execute_tool("logging_probe",
                         {"password": "hunter2", "note": "Bearer abc.def.ghi-secret"},
                         request_id="req-123")
        record = [r for r in caplog.records if r.getMessage() == "tool_call"][0]
        fields = record.fields
        assert fields["tool"] == "logging_probe" and fields["status"] == "ok"
        assert fields["request_id"] == "req-123" and fields["duration_ms"] >= 0
        assert "hunter2" not in str(fields) and "abc.def.ghi-secret" not in str(fields)
    finally:
        REGISTRY.pop("logging_probe", None)
