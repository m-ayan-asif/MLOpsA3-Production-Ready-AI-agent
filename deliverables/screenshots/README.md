# Screenshot evidence

## Generated evidence summaries

The numbered PNGs `01_live_stack_status.png` through `04_failure_investigations.png` are programmatically rendered from captured `/chat` JSON and live service status. `02_multi_tool_conversation.png` is generated from the exact mandatory prompt and successful four-system trace in `../conversations/01_multi_tool.json`. These summaries are not pixel captures of the Airflow, MLflow, browser, or terminal UI.

## Authentic application and terminal captures

- `airflowSuccess.png` — successful Airflow DAG run and five task states.
- `mlflowRunMetrics.png` — MLflow run metrics, quality-gate tag, and dataset/DVC/Git provenance tags.
- `kubernetesReady.png` — inference deployment replicas and Service endpoints.
- `dvcClean.png` — DVC status reporting that data and pipelines are up to date.
- `chatApiRequest.png`, `Responses1.png`, and `Responses2.png` — Swagger request, successful response/tool trace, and response schema.
- `serviceUptimeConfirmation.png` — Docker Compose services and Kubernetes readiness.
