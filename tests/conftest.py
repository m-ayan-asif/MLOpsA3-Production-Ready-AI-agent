from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import agent.tools  # noqa: F401
from agent.llm import LLMTurn, ToolCall
from agent.tools import airflow_tools
from agent.tools.base import REGISTRY


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("K8S_ALLOWED_NAMESPACES", "mlops-agent")
    monkeypatch.setenv("APPROVED_DAG_IDS", "ml_training_pipeline")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "super-secret-pw")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "5")
    airflow_tools._token_cache.update(token=None, expires=0.0)


class FakeLLM:
    """Scripted LLM: each script item is a list of ToolCalls or a final-text string."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.seen_tool_payloads: list[Any] = []

    def generate(self, system, messages, tools):
        last = messages[-1]
        if last["role"] == "tool":
            self.seen_tool_payloads.append(last["results"])
        step = self.script.pop(0)
        if isinstance(step, str):
            return LLMTurn(text=step)
        return LLMTurn(tool_calls=step)


def calls(*names: str) -> list[ToolCall]:
    return [ToolCall(n, {}) for n in names]


@pytest.fixture
def stub_tool(monkeypatch):
    """Replace a registered tool's implementation (keeps its schema and permissions)."""

    def _stub(name: str, func):
        monkeypatch.setitem(REGISTRY, name, replace(REGISTRY[name], func=func))

    return _stub


def make_run(run_id, f1, status="FINISHED", tags=None, end_ms=1_700_000_100_000):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, run_name=run_id, experiment_id="1", status=status,
                             start_time=1_700_000_000_000, end_time=end_ms),
        data=SimpleNamespace(metrics={"f1": f1}, params={"n_estimators": "50"}, tags=tags or {}),
    )


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload
