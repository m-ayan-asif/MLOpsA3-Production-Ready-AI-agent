"""Multi-tool agent loop: LLM selects tools, every call is validated/gated/logged, and the
final answer is wrapped with deterministic evidence and failure notices."""
from __future__ import annotations

import uuid
from typing import Any

import agent.tools  # noqa: F401  (registers tools)
from agent.llm import LLMClient
from agent.logging_utils import log_event
from agent.schemas import AgentResult, ToolResult, ToolTrace
from agent.tools.base import REGISTRY, execute_tool

SYSTEM_PROMPT = """You are an MLOps assistant for one team. You can inspect DVC datasets, MLflow \
experiments, Airflow pipelines and a Kubernetes deployment, only through the provided tools.

Rules:
1. Every operational claim (versions, metrics, run status, pod health) must come from a tool \
result in this conversation. Never answer from memory or guess values.
2. If a tool returns an error, timeout, denial or confirmation_required, say exactly that. Do not \
fill the gap. Mark the affected facts as UNKNOWN.
3. Pick the tools the question needs; for broad questions call several. Run independent calls in \
the same step.
4. The highest metric does not by itself make a model production-ready. Report the readiness \
checks and any that failed.
5. Never start retraining, restart workloads or change data unless the user explicitly asked for \
that action. Those tools need human confirmation that the system handles. If a tool returns \
confirmation_required, tell the user what would change and ask them to confirm.
6. Recommend retraining only with evidence: an explicit freshness policy being exceeded, a failed \
quality gate, or degradation results the user supplied. Use assess_retraining_need for that.
7. Format the final answer with three headed sections:
   **Known facts** (each tied to the tool that produced it)
   **Uncertainties** (unavailable tools, missing data, assumptions)
   **Recommended actions** (what a human should do next; actions are never taken automatically).
8. For a request about the best recent model, its dataset version, deployment status and retraining \
need, gather evidence from MLflow, DVC, Kubernetes and Airflow. Retrieve DVC provenance tied to the \
selected MLflow run; current workspace dataset state is not a substitute. Check the actual state of \
the latest relevant Airflow training run; listing DAG definitions is not a status check. If any of \
these results is missing, continue gathering evidence before answering. MLflow's DVC tags alone do \
not count as a DVC lookup. This assessment is read-only.
Be concise."""

MAX_STEPS = 8
_MANDATORY_ASSESSMENT = (
    "best recent model", "dataset version", "deployment status", "retraining",
)
_MANDATORY_EVIDENCE = (
    ("mlflow_best_run", "a best-run evaluation from MLflow"),
    ("dvc_get_run_dataset", "DVC provenance for the selected run"),
    ("k8s_get_deployed_version", "Kubernetes deployment status"),
    ("airflow_latest_run_summary", "the latest Airflow training-run status"),
    ("assess_retraining_need", "a retraining evidence assessment"),
)


def _missing_mandatory_evidence(message: str, results: list[ToolResult]) -> list[str]:
    normalized = " ".join(message.casefold().split())
    if not all(marker in normalized for marker in _MANDATORY_ASSESSMENT):
        return []
    successful = {result.tool for result in results if result.ok}
    return [label for tool, label in _MANDATORY_EVIDENCE if tool not in successful]


class Orchestrator:
    def __init__(self, llm: LLMClient, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.max_steps = max_steps

    def run(self, message: str, *, confirm: bool = False,
            request_id: str | None = None) -> AgentResult:
        rid = request_id or uuid.uuid4().hex[:12]
        specs = list(REGISTRY.values())
        messages: list[dict[str, Any]] = [{"role": "user", "text": message}]
        results: list[ToolResult] = []
        final: str | None = None
        steps = 0

        log_event("agent_request", request_id=rid, confirm=confirm, message_chars=len(message))
        for step in range(1, self.max_steps + 1):
            steps = step
            turn = self.llm.generate(SYSTEM_PROMPT, messages, specs)
            messages.append({"role": "model", "text": turn.text, "raw": turn.raw})
            if not turn.tool_calls:
                missing = _missing_mandatory_evidence(message, results)
                if missing and step < self.max_steps:
                    messages.append({
                        "role": "user",
                        "text": "Your assessment is incomplete. Before giving a final answer, "
                                "obtain successful evidence for: " + "; ".join(missing)
                                + ". Use the available tools and do not substitute metadata "
                                "or current workspace state for the requested evidence.",
                    })
                    continue
                if missing:
                    final = "I could not complete this assessment because I did not obtain: " \
                            + "; ".join(missing) + "."
                    break
                final = turn.text or ""
                break
            batch: list[tuple[str, dict[str, Any]]] = []
            for call in turn.tool_calls:
                result = execute_tool(call.name, call.args, request_id=rid, confirmed=confirm)
                results.append(result)
                batch.append((call.name, result.llm_payload()))
            messages.append({"role": "tool", "results": batch})
        if final is None:
            final = "I stopped before finishing: the step limit was reached."

        failures = [f"{r.tool}: {r.status}" + (f" ({r.error})" if r.error else "")
                    for r in results if not r.ok]
        answer = self._grounded(final, results, failures)
        log_event("agent_response", request_id=rid, steps=steps, tool_calls=len(results),
                  failures=len(failures))
        return AgentResult(
            request_id=rid, answer=answer, steps=steps, tool_failures=failures,
            tool_trace=[ToolTrace(tool=r.tool, source=r.source, status=r.status,
                                  duration_ms=r.duration_ms, error=r.error) for r in results])

    @staticmethod
    def _grounded(final: str, results: list[ToolResult], failures: list[str]) -> str:
        notices = []
        if not results:
            notices.append("No tools were called, so nothing below is backed by operational "
                           "evidence.")
        if failures:
            notices.append("These tool calls did not succeed, so related facts are NOT verified: "
                           + "; ".join(failures))
        if not notices:
            return final
        return "\u26a0\ufe0f " + " ".join(notices) + "\n\n" + final
