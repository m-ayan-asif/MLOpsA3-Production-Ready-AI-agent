"""Inference service deployed on Kubernetes. Liveness != readiness on purpose:
a pod without a model stays alive (no restart loop) but reports NOT READY."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("inference")
N_FEATURES = 30
state: dict[str, object] = {"model": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    path = os.getenv("MODEL_PATH", "/models/model.joblib")
    try:
        import joblib

        state["model"], state["error"] = joblib.load(path), None
        log.info("model loaded from %s", path)
    except Exception as exc:  # noqa: BLE001
        state["model"], state["error"] = None, f"{type(exc).__name__}: {exc}"
        log.error("model load failed path=%s error=%s", path, state["error"])
    yield


app = FastAPI(title="Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready",
                                                      "reason": state["error"]})
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"app_version": os.getenv("APP_VERSION", "unknown"),
            "model_version": os.getenv("MODEL_VERSION", "unknown")}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, int]:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"prediction": int(model.predict([req.features])[0])}
