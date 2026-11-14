# Architecture

```
User ─► FastAPI (agent/main.py: validation, /chat, /tools)
          └─► Orchestrator (agent/orchestrator.py)
                ├─ Gemini (agent/llm.py)   manual function calling, automatic calling disabled
                └─ execute_tool (agent/tools/base.py)
                     schema validation → permission/confirmation gate → timeout → structured log
                       ├─ dvc_tools.py        allowlisted git/dvc subprocess + MLflow provenance tags
                       ├─ mlflow_tools.py     MlflowClient (runs, metrics, registry)
                       ├─ airflow_tools.py    Airflow 3 REST (JWT via /auth/token, /api/v2)
                       ├─ kubernetes_tools.py kubernetes client, approved namespaces only
                       └─ policy_tools.py     deterministic retraining-evidence check (read-only)
        └─► Response = LLM answer + deterministic tool trace + failure notice
```

## Design decisions
- **Provenance lives in MLflow.** `training/steps.py` tags every run with `dvc.md5`, `git.commit` and
  `dataset.version` (from the `dataset-*` Git tag). "Which dataset did experiment N use?" is answered
  from that record, never inferred.
- **The LLM cannot authorize anything.** State-changing tools (`airflow_trigger_dag`,
  `k8s_restart_deployment`, `dvc_checkout_version`) take a `confirm` argument that is hidden from the LLM
  schema and injected by `execute_tool` from the API request's `confirm` flag. Without it, the tool returns
  `confirmation_required` and nothing happens.
- **Grounding is enforced outside the prompt.** Tool errors reach the LLM as `data: null`; the final answer
  is prefixed with a warning listing every failed tool, and an answer produced with zero tool calls is
  flagged as unbacked. The trace of real tool calls is returned separately from the LLM's text.
- **Liveness vs readiness.** The inference pod stays alive without a model (no restart loop) but reports
  not-ready, so the Service has no endpoints and the failure is diagnosable from probes, events and logs.
- **Promotion is human.** The DAG's quality gate only tags the run and sets the `candidate` alias. A failed
  validation or gate raises `AirflowFailException` (no retries, graph stops).

## Tool permissions
| Tool | System | Access | Guard |
|---|---|---|---|
| dvc_list_datasets, dvc_get_dataset_version, dvc_compare_versions, dvc_detect_changes, dvc_explain_reproduce, dvc_get_run_dataset | DVC/Git/MLflow | read | command allowlist, rev/path regex |
| dvc_checkout_version | DVC/Git | **write (workspace)** | confirm flag |
| mlflow_list_experiments, mlflow_list_runs, mlflow_get_run, mlflow_best_run, mlflow_compare_runs, mlflow_compare_experiments, mlflow_model_registry | MLflow | read | name regex |
| airflow_list_dags, airflow_get_dag_runs, airflow_latest_run_summary, airflow_get_failed_tasks, airflow_get_task_info | Airflow | read | JWT auth |
| airflow_trigger_dag | Airflow | **write** | confirm flag, approved DAG list, conf key allowlist |
| k8s_list_workloads, k8s_describe_pod, k8s_get_pod_logs, k8s_list_services, k8s_find_unhealthy_pods, k8s_get_deployed_version | Kubernetes | read | approved namespaces, RBAC Role |
| k8s_restart_deployment | Kubernetes | **write** | namespace check, then confirm flag |
| assess_retraining_need | MLflow + Airflow | read | never triggers |
