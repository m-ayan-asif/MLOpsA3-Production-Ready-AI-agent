"""Tool registry: schema validation, timeouts, permission gating, structured logging."""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator

from agent.config import get_settings
from agent.logging_utils import log_event, redact
from agent.schemas import ToolResult

MUTATING_NOTE = (
    " STATE-CHANGING: requires explicit human confirmation, which the system injects. "
    "If the result is 'confirmation_required', tell the user what would change and ask them "
    "to confirm; never claim the action happened."
)


class ToolError(Exception):
    """Expected failure with a machine-readable code (denied, not_found, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    source: str
    mutating: bool = False

    def llm_schema(self) -> dict[str, Any]:
        """Schema shown to the LLM: no `confirm` (system-controlled), no additionalProperties."""
        props = {k: v for k, v in self.parameters["properties"].items() if k != "confirm"}
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if self.parameters.get("required"):
            schema["required"] = self.parameters["required"]
        return schema


REGISTRY: dict[str, ToolSpec] = {}


def tool(*, source: str, description: str, properties: dict[str, Any] | None = None,
         required: tuple[str, ...] = (), mutating: bool = False):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        props = dict(properties or {})
        if mutating:
            props["confirm"] = {"type": "boolean"}
        schema = {"type": "object", "properties": props, "required": list(required),
                  "additionalProperties": False}
        desc = description + (MUTATING_NOTE if mutating else "")
        REGISTRY[func.__name__] = ToolSpec(func.__name__, desc, schema, func, source, mutating)
        return func

    return decorator


def _coerce(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """LLMs often send 5.0 for integers; normalise after validation."""
    out = dict(args)
    for key, value in args.items():
        declared = spec.parameters["properties"].get(key, {}).get("type")
        if declared == "integer" and isinstance(value, float):
            out[key] = int(value)
    return out


def execute_tool(name: str, args: dict[str, Any] | None, *, request_id: str,
                 confirmed: bool = False, timeout: float | None = None) -> ToolResult:
    started = time.perf_counter()
    spec = REGISTRY.get(name)
    source = spec.source if spec else "unknown"

    def finish(status: str, *, data: Any = None, error: str | None = None,
               code: str | None = None) -> ToolResult:
        duration = round((time.perf_counter() - started) * 1000, 1)
        result = ToolResult(tool=name, source=source, status=status, data=data, error=error,
                            error_code=code, duration_ms=duration, request_id=request_id)
        log_event("tool_call", tool=name, source=source, status=status, duration_ms=duration,
                  request_id=request_id, error_code=code, args=args or {})
        return result

    if spec is None:
        return finish("unknown_tool", error=f"Unknown tool '{name}'", code="unknown_tool")

    call_args = dict(args or {})
    if spec.mutating:
        call_args["confirm"] = bool(confirmed)  # overrides anything the LLM sent
    problems = sorted(Draft202012Validator(spec.parameters).iter_errors(call_args),
                      key=lambda e: list(e.path))
    if problems:
        message = "; ".join(p.message for p in problems)[:400]
        return finish("invalid_arguments", error=message, code="invalid_arguments")
    call_args = _coerce(spec, call_args)

    limit = timeout or get_settings().tool_timeout
    pool = cf.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(spec.func, **call_args)
    try:
        data = future.result(timeout=limit)
    except cf.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        return finish("timeout", error=f"Tool did not respond within {limit}s", code="timeout")
    except ToolError as exc:
        pool.shutdown(wait=False)
        status = exc.code if exc.code in ("denied", "confirmation_required") else "error"
        return finish(status, error=exc.message, code=exc.code)
    except Exception as exc:  # noqa: BLE001 - any backend failure becomes a structured error
        pool.shutdown(wait=False)
        return finish("error", error=str(redact(f"{type(exc).__name__}: {exc}"))[:300],
                      code="tool_failure")
    pool.shutdown(wait=False)
    return finish("ok", data=data)
