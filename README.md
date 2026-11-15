# Assignment 03 — Production-Ready MLOps AI Agent

A Gemini-powered agent that answers questions about, and (with confirmation) operates, a real
DVC + MLflow + Airflow 3 + Kubernetes stack. See [architecture.md](architecture.md) for design and
tool permissions.

## Repository layout

The project follows the recommended Assignment 03 repository layout. Application code is grouped
under `agent/`, `api/`, `dags/`, and `training/`; deployment manifests are under `kubernetes/`;
tests and CI are under `tests/` and `.github/workflows/`; and authentic evaluation captures are in
the repository-root [screenshots/](screenshots/README.md) directory. Additional setup scripts,
Docker configuration, and the [submission evidence package](deliverables/README.md) are included
to support local operation and assessment.

## Submission evidence

The assessment report, failure investigations, conversation traces, and screenshot evidence are
organized under [deliverables/](deliverables/README.md). Authentic UI/terminal captures and
generated evidence summaries are distinguished in [deliverables/screenshots/](deliverables/screenshots/README.md).
The authentic captures are also copied into the required repository-root `screenshots/` folder.

## 0. Prerequisites
Python 3.10+, Docker, `kind`, `kubectl`, Git, a Gemini API key.

## 1. Python environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mlops,dev]"
cp .env.example .env        # fill GEMINI_API_KEY and AIRFLOW_PASSWORD; never commit .env
```

Put your Gemini API key in `GEMINI_API_KEY` in the project-root `.env` file. The agent loads
this file automatically at startup; `.env` is git-ignored. The default model is
`gemini-3.1-flash-lite` and can be overridden with `GEMINI_MODEL`.

## 2. Versioned data (DVC)
```bash
./scripts/bootstrap_data.sh   # creates dataset-v1.0 and dataset-v1.1 tags + local DVC remote
```

## 3. MLflow + Airflow
```bash
docker compose up -d --build   # MLflow :5000, Airflow 3 :8080 (login: AIRFLOW_USERNAME / AIRFLOW_PASSWORD)
./scripts/seed_runs.sh         # 5 tracked training runs with provenance tags
python -m training.train --export models/model.joblib   # model baked into the inference image
```
Trigger the DAG from the Airflow UI or through the agent (section 6).

## 4. Kubernetes
```bash
./scripts/k8s_up.sh            # kind cluster, image build/load, apply manifests, wait for rollout
kubectl apply -f kubernetes/rbac.yaml   # least-privilege ServiceAccount for the agent
```

## 5. Run the agent
```bash
uvicorn agent.main:app --port 8000
curl -s localhost:8000/tools | jq length
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}' | jq -r .answer
```
State-changing requests need the human to confirm explicitly: add `"confirm": true` only after the
agent has told you what would change.
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Trigger the model training pipeline and report whether it succeeded.","confirm":true}'
```
The response contains `answer`, `tool_trace` (every real tool call, status and duration) and
`tool_failures`.

## 6. Demo scenarios
| Scenario | Prompt |
|---|---|
| Dataset provenance | "Which dataset version was used for the latest run, and how do I reproduce it?" |
| Model selection | "Compare the last five training runs and recommend the best candidate by F1." |
| Pipeline execution | "Trigger the training pipeline and report whether it succeeded." (with `confirm`) |
| Deployment diagnosis | "Why is my prediction API unavailable?" |
| Multi-tool | the mandatory scenario in section 5 |

## 7. Failure injection
```bash
./scripts/failures.sh k8s-bad-image        # ErrImagePull / ImagePullBackOff on new pods
./scripts/failures.sh k8s-bad-model-path   # pods Running but not Ready (readiness 503), no endpoints
./scripts/failures.sh dag-corrupt-data     # DVC hash mismatch -> validate_dataset task fails
./scripts/failures.sh restore
```
Ask: "Why is my prediction API unavailable?" / "Did the latest training run fail? Why, and has the
tracked data changed?" and keep the `tool_trace` as report evidence.

## 8. Tests and lint (all external systems mocked; this is what CI runs)
```bash
ruff check . && pytest -q
```

## Environment variables
See `.env.example`. The agent loads the project-root `.env` file; secrets are not committed, and
logs redact tokens, passwords and keys.

## Known limitations
- Tool timeouts stop waiting but cannot kill a blocked thread; the call finishes in the background.
- `dvc_detect_changes` parses `dvc data status --json`; field names can differ across DVC versions.
- Gemini model name is configurable (`GEMINI_MODEL`); change it if the default is retired.
