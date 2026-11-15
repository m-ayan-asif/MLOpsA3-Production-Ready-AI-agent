"""LLM abstraction. The orchestrator only sees LLMClient; Gemini is one implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import get_settings
from agent.tools.base import ToolSpec


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMTurn:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native content, replayed verbatim on the next turn


class LLMError(RuntimeError):
    pass


MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 0.5


def _is_transient_error(exc: Exception) -> bool:
    status = getattr(exc, "code", None)
    if status is None:
        status = getattr(exc, "status", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return status in (408, 429) or 500 <= status < 600


class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMTurn: ...


class GeminiClient:
    """Manual function calling (automatic calling disabled) so every call goes through
    our validation, permission and logging layer."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        settings = get_settings()
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self._model = model or settings.gemini_model

    def generate(self, system: str, messages: list[dict[str, Any]],
                 tools: list[ToolSpec]) -> LLMTurn:
        from google.genai import types

        declarations = [types.FunctionDeclaration(name=t.name, description=t.description,
                                                  parameters_json_schema=t.llm_schema())
                        for t in tools]
        contents: list[Any] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user",
                                              parts=[types.Part.from_text(text=m["text"])]))
            elif m["role"] == "model":
                contents.append(m["raw"] if m.get("raw") is not None else types.Content(
                    role="model", parts=[types.Part.from_text(text=m.get("text") or "")]))
            else:  # tool results
                parts = [types.Part.from_function_response(name=name, response=payload)
                         for name, payload in m["results"]]
                contents.append(types.Content(role="user", parts=parts))
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.1,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_RETRIES or not _is_transient_error(exc):
                    raise LLMError(f"Gemini request failed: {type(exc).__name__}") from exc
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))
        if not response.candidates or not response.candidates[0].content:
            raise LLMError("Gemini returned no content")
        content = response.candidates[0].content
        parts = content.parts or []
        calls = [ToolCall(p.function_call.name, dict(p.function_call.args or {}))
                 for p in parts if p.function_call]
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
        return LLMTurn(text=text or None, tool_calls=calls, raw=content)
