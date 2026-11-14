"""Structured inputs/outputs shared by tools, orchestrator and API."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal[
    "ok", "error", "denied", "confirmation_required", "timeout", "invalid_arguments", "unknown_tool"
]


class ToolResult(BaseModel):
    tool: str
    source: str
    status: Status
    data: Any = None
    error: str | None = None
    error_code: str | None = None
    duration_ms: float
    request_id: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def llm_payload(self, limit: int = 12000) -> dict[str, Any]:
        """What the LLM sees. Errors carry no data, so there is nothing to 'fill in'."""
        payload = {"status": self.status, "error": self.error, "data": self.data}
        text = json.dumps(payload, default=str)
        if len(text) > limit:
            return {"status": self.status, "error": self.error, "data_truncated": True,
                    "data_preview": text[:limit]}
        return json.loads(text)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Out-of-band confirmation for state-changing tools. Set by the client only when
    # the human has explicitly agreed; the LLM can never set it.
    confirm: bool = False


class ToolTrace(BaseModel):
    tool: str
    source: str
    status: str
    duration_ms: float
    error: str | None = None


class AgentResult(BaseModel):
    request_id: str
    answer: str
    steps: int
    tool_trace: list[ToolTrace]
    tool_failures: list[str]
