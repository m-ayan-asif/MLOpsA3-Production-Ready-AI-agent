"""Training DAG: validate -> preprocess -> train -> evaluate/log to MLflow -> quality gate.

Validation and gate failures raise AirflowFailException (no pointless retries) and stop the
graph, so nothing downstream is promoted. Transient errors (e.g. MLflow briefly down) use
the default retries.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from airflow.sdk import dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # older 3.x layouts
    from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)
REPO = os.getenv("REPO_ROOT", "/opt/airflow/repo")
WORK = os.getenv("WORK_DIR", "/opt/airflow/work")
DATASET = os.path.join(REPO, "data", "dataset.csv")


def _params() -> dict[str, Any]:
    p = get_current_context()["params"]
    return {"n_estimators": int(p["n_estimators"]),
            "max_depth": None if p["max_depth"] in (None, 0) else int(p["max_depth"]),
            "seed": int(p["seed"])}


@dag(
    dag_id="ml_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
    params={"n_estimators": 100, "max_depth": 0, "seed": 42, "f1_threshold": 0.93},
    tags=["mlops", "assignment-03"],
)
def ml_training_pipeline():
    @task
    def validate_dataset() -> dict[str, Any]:
        from training import steps

        try:
            meta = steps.validate_dataset(DATASET)
        except ValueError as exc:
            log.error("dataset validation failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc
        log.info("dataset valid: %s", meta)
        return meta

    @task
    def preprocess(meta: dict[str, Any]) -> dict[str, Any]:
        from training import steps

        ctx = get_current_context()
        work = os.path.join(WORK, ctx["run_id"].replace(":", "_").replace("+", "_"))
        prep = steps.preprocess(meta, work, _params()["seed"])
        log.info("preprocessed: %s train / %s test rows", prep["train_rows"], prep["test_rows"])
        return prep

    @task
    def train_model(prep: dict[str, Any]) -> dict[str, Any]:
        from training import steps

        trained = steps.train_model(prep, _params())
        log.info("model trained with %s", trained["params"])
        return trained

    @task
    def evaluate_and_log(trained: dict[str, Any]) -> dict[str, Any]:
        from training import steps

        os.environ["PIPELINE_NAME"] = "airflow:ml_training_pipeline"
        result = steps.evaluate_and_log(trained, run_name=get_current_context()["run_id"][-24:])
        log.info("logged MLflow run %s metrics=%s", result["run_id"], result["metrics"])
        return result

    @task
    def quality_gate(result: dict[str, Any]) -> dict[str, Any]:
        from training import steps

        threshold = float(get_current_context()["params"]["f1_threshold"])
        try:
            outcome = steps.quality_gate(result, threshold)
        except ValueError as exc:
            log.error("%s", exc)
            raise AirflowFailException(str(exc)) from exc
        log.info("quality gate passed (f1 >= %s); registered as 'candidate'", threshold)
        return outcome

    quality_gate(evaluate_and_log(train_model(preprocess(validate_dataset()))))


ml_training_pipeline()
