# Failure Investigations

All three documented failure-injection scenarios were executed against the live stack and restored afterward. These notes separate direct observations from what the agent reported.

## 1. Kubernetes image pull failure

**Injection:** Updated the `inference-api` Deployment image to `mlops-inference:does-not-exist`.

**Observation:** A new pod remained Pending with `ErrImagePull` (and then `ImagePullBackOff`). Two old pods using `mlops-inference:1.0.0` remained healthy during the rolling rollout, so the Service still had capacity. The failure demonstrated rollout protection rather than a total outage.

**Agent diagnosis:** The agent called `k8s_list_workloads` and `k8s_find_unhealthy_pods`, identified the invalid image tag and failing pod, and recommended restoring the valid tag. Both tool calls succeeded.

**Recovery:** Restored image `mlops-inference:1.0.0`, restarted the Deployment, and waited for rollout completion. Final state was 2/2 ready replicas.

## 2. Kubernetes readiness failure from a bad model path

**Injection:** Changed `inference-config` so `MODEL_PATH=/models/missing.joblib`, restarted the Deployment, and replaced the old replicas so the test represented the documented no-endpoints failure rather than a mixed rolling update.

**Observation:** New pods remained in Running phase, but both were unready because the readiness probe returned HTTP 503. The Service had zero ready endpoints. This is distinct from liveness: the process remains alive for diagnosis, but receives no Service traffic.

**Agent diagnosis:** The agent identified 0/2 ready replicas and the unhealthy pods/readiness failure. In an earlier diagnosis call it also retrieved the ConfigMap value and identified the missing model path. In the zero-endpoint run, it did not inspect the ConfigMap and appropriately marked the internal cause unknown rather than inventing it.

**Recovery:** Reapplied `kubernetes/configmap.yaml`, restored two replicas, restarted, and waited for rollout. Final state returned to 2/2 ready with two endpoints.

## 3. DVC data corruption and Airflow validation failure

**Injection:** Appended `corrupt,row` to `data/dataset.csv`, changing its actual MD5 to `ac8cabd3dc372c7ef3c1f92695bc013d`; the DVC record remained `fca8d29f55d3bb4293b10d4935173c34`. Triggered `ml_training_pipeline`.

**Observation:** The DAG failed at `validate_dataset`. `preprocess`, `train_model`, `evaluate_and_log`, and `quality_gate` were upstream-failed; no training was performed from the corrupted dataset. The exact run and task log are preserved in [failure_evidence/airflow_dvc_corruption.json](./failure_evidence/airflow_dvc_corruption.json).

**Agent diagnosis:** While the corruption was still present, a saved live conversation identified the Airflow hash mismatch and DVC's changed-data status. A later follow-up conversation was run after restoration and therefore correctly observed a clean current workspace; because it did not know the injection/restore timeline, it called out a temporal discrepancy. See [conversations/05_failure_investigation.json](./conversations/05_failure_investigation.json).

**Recovery:** `dvc checkout` initially refused to overwrite the unsaved corruption without confirmation. Restored the tracked output with `dvc checkout --force data/dataset.csv.dvc` and verified `dvc status` reported “Data and pipelines are up to date.” The forced checkout was narrowly scoped to the named DVC-managed dataset.

## Additional live Airflow setup defects

These were not assignment failure injections, but were uncovered by running the real DAG:

1. The Airflow task user could not create the run-specific directory under `/opt/airflow/work` (`PermissionError`). The Airflow image now creates the directory with `airflow:root` ownership, and the existing named work volume was corrected. A write test passed.
2. The model-logging task failed with `ModuleNotFoundError: No module named 'skops'`. The Airflow image now installs `skops`.

After both fixes, a full Airflow run completed all five stages successfully, and its MLflow run was tagged with dataset MD5, Git commit, and `dataset-v1.1`.
