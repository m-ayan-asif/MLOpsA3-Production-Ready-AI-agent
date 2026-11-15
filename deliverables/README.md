# Verification deliverables

- [Verification report](./report.md) — implementation, rubric evidence, live results, limitations, and remaining submission work.
- [Failure investigations](./failure-investigations.md) — injected incidents, evidence, diagnosis, and recovery.
- [Five required conversation examples](./conversations/) — request/response JSON with live tool traces.
- [Mandatory four-system conversation](./conversations/01_multi_tool.json) — one successful request correlating MLflow, DVC, Kubernetes, and Airflow.
- [Confirmed Airflow trigger](./conversations/06_pipeline_execution.json) and [final status check](./conversations/06_pipeline_execution_result.json) — the agent triggered one run with `confirm: true`; the read-only follow-up verified all five tasks succeeded.
- [Preserved multi-tool attempts](./conversations/attempts/) — includes the initial Gemini 502, a partial attempt, and the previous narrower success.
- [Earlier live-state snapshot](./live_status.json) — the latest Airflow execution is recorded separately in the pipeline conversation evidence above.
- [Airflow corruption evidence](./failure_evidence/airflow_dvc_corruption.json) — failed DAG run, task states, and validation log.
- [Screenshots](./screenshots/) — generated evidence summaries and authentic Airflow, MLflow, Kubernetes, DVC, and agent/API captures. The Swagger chat captures show a narrower Kubernetes/MLflow request, not the mandatory four-system scenario.
- [Hosted GitHub Actions run](https://github.com/m-ayan-asif/MLOpsA3-Production-Ready-AI-agent/actions/runs/43694353648) — passed for the published initial commit `ca16246`; it runs Ruff and pytest.

The report and conversation evidence updates are local and have not been pushed. The hosted CI result above applies to `ca16246`, not to those follow-up changes.
