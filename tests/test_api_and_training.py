import hashlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.main import app, get_orchestrator
from agent.orchestrator import Orchestrator
from api import main as inference
from tests.conftest import FakeLLM, calls
from training import steps


def test_agent_chat_endpoint_validates_input_and_returns_trace(stub_tool):
    stub_tool("mlflow_list_experiments", lambda: {"experiments": []})
    app.dependency_overrides[get_orchestrator] = lambda: Orchestrator(
        FakeLLM([calls("mlflow_list_experiments"), "No experiments."]))
    try:
        client = TestClient(app)
        assert client.post("/chat", json={"message": ""}).status_code == 422
        body = client.post("/chat", json={"message": "list experiments"}).json()
        assert body["tool_trace"][0]["tool"] == "mlflow_list_experiments"
        assert body["request_id"]
    finally:
        app.dependency_overrides.clear()


def test_inference_service_not_ready_without_model_but_alive():
    inference.state.update(model=None, error="FileNotFoundError: /models/missing.joblib")
    client = TestClient(inference.app)
    assert client.get("/healthz").status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 503 and "missing.joblib" in ready.json()["reason"]
    assert client.post("/predict", json={"features": [0.0] * 30}).status_code == 503


def test_training_provenance_reads_recorded_dvc_hash(tmp_path):
    data = tmp_path / "dataset.csv"
    data.write_text("a,target\n1,0\n")
    (tmp_path / "dataset.csv.dvc").write_text("outs:\n- md5: 1234abcd\n  path: dataset.csv\n")
    assert steps.dvc_record(data)["md5"] == "1234abcd"
    assert steps.file_md5(data) == hashlib.md5(data.read_bytes()).hexdigest()  # noqa: S324


def test_training_provenance_resolves_git_from_dataset_directory(monkeypatch, tmp_path):
    data = tmp_path / "repo" / "data" / "dataset.csv"
    data.parent.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs["cwd"])
        stdout = {
            "rev-parse": str(tmp_path / "repo"),
            "log": "commit123",
            "tag": "dataset-v1.1",
        }[command[1]]
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(steps.subprocess, "run", fake_run)
    monkeypatch.setattr(steps, "dvc_record", lambda _: {"md5": "1234abcd"})

    result = steps.read_provenance(data)

    assert result["git.commit"] == "commit123"
    assert result["dataset.version"] == "dataset-v1.1"
    assert calls == [data.parent, tmp_path / "repo", tmp_path / "repo"]
