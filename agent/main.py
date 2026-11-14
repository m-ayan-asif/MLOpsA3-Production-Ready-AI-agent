"""FastAPI interface for the agent: validation in, evidence-based response out."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from agent.config import get_settings
from agent.llm import GeminiClient, LLMError
from agent.orchestrator import Orchestrator
from agent.schemas import AgentResult, ChatRequest
from agent.tools.base import REGISTRY

app = FastAPI(title="MLOps Agent", version="0.1.0")


def get_orchestrator() -> Orchestrator:
    if not get_settings().gemini_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    return Orchestrator(GeminiClient())


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list[dict[str, object]]:
    return [{"name": t.name, "source": t.source, "mutating": t.mutating,
             "description": t.description} for t in REGISTRY.values()]


@app.post("/chat", response_model=AgentResult)
def chat(req: ChatRequest, orchestrator: Orchestrator = Depends(get_orchestrator)) -> AgentResult:
    try:
        return orchestrator.run(req.message, confirm=req.confirm)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
