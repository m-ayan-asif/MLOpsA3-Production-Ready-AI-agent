# Verification deliverables

- [Verification report](./report.md) — live results, test evidence, caveats, submission checklist, and an evidence map for the supplied 100-mark assessment rubric.
- [Failure investigations](./failure-investigations.md) — injected incidents, direct evidence, agent diagnosis, and recovery.
- [Five conversation examples](./conversations/) — request/response JSON with real tool traces.
- [Initial Gemini error attempt](./conversations/attempts/01_multi_tool_initial_502.json) — preserved separately; the successful retry is conversation 01.
- [Final live state](./live_status.json) — service health, cluster state, DVC status, and recent runs.
- [Airflow corruption evidence](./failure_evidence/airflow_dvc_corruption.json) — failed DAG run, task states, and validation log.
- [Screenshots](./screenshots/) — both generated evidence summaries and authentic Airflow, MLflow, Kubernetes, DVC, and agent/API captures.
- A verified GitHub remote/hosted CI run and the live demonstration/oral defense remain submission follow-ups.

The project source changes were not committed as part of this evidence package.
