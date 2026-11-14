"""Environment-driven settings. Read on every call so tests can patch the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str
    mlflow_experiment: str
    airflow_base_url: str
    airflow_username: str
    airflow_password: str = field(repr=False)
    approved_dag_ids: tuple[str, ...]
    k8s_allowed_namespaces: tuple[str, ...]
    k8s_app_label: str
    k8s_deployment: str
    k8s_configmap: str
    dvc_repo_path: str
    gemini_api_key: str = field(repr=False)
    gemini_model: str
    tool_timeout: float
    http_timeout: float
    model_freshness_days: int
    quality_gate_f1: float


def get_settings() -> Settings:
    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "mlops-agent-demo"),
        airflow_base_url=os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/"),
        airflow_username=os.getenv("AIRFLOW_USERNAME", "admin"),
        airflow_password=os.getenv("AIRFLOW_PASSWORD", ""),
        approved_dag_ids=_csv("APPROVED_DAG_IDS", "ml_training_pipeline"),
        k8s_allowed_namespaces=_csv("K8S_ALLOWED_NAMESPACES", "mlops-agent"),
        k8s_app_label=os.getenv("K8S_APP_LABEL", "app.kubernetes.io/name=inference-api"),
        k8s_deployment=os.getenv("K8S_DEPLOYMENT", "inference-api"),
        k8s_configmap=os.getenv("K8S_CONFIGMAP", "inference-config"),
        dvc_repo_path=os.getenv("DVC_REPO_PATH", "."),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        tool_timeout=float(os.getenv("TOOL_TIMEOUT_SECONDS", "30")),
        http_timeout=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
        model_freshness_days=int(os.getenv("MODEL_FRESHNESS_DAYS", "30")),
        quality_gate_f1=float(os.getenv("QUALITY_GATE_F1", "0.93")),
    )
