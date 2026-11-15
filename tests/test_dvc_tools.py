import json

import pytest

from agent.tools import dvc_tools
from agent.tools.base import ToolError, execute_tool

DVC_A = "outs:\n- md5: aaaa\n  size: 100\n  path: dataset.csv\n"
DVC_B = "outs:\n- md5: bbbb\n  size: 160\n  path: dataset.csv\n"


def test_disallowed_commands_never_reach_subprocess(monkeypatch):
    monkeypatch.setattr(dvc_tools.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("executed")))
    for cmd in (["rm", "-rf", "/"], ["dvc", "gc", "-f"], ["git", "push", "origin"],
                ["git", "checkout", "main"]):
        with pytest.raises(ToolError) as exc:
            dvc_tools._run(cmd)
        assert exc.value.code == "denied"


def test_version_includes_hash_commit_and_tag(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/dataset.csv.dvc").write_text(DVC_B)
    monkeypatch.setenv("DVC_REPO_PATH", str(tmp_path))

    def fake_run(cmd):
        if cmd[:2] == ["git", "log"]:
            return "a4c92e1deadbeef\x1f2026-10-01T10:00:00+00:00\x1fdata: dataset v1.1\n"
        if cmd[:2] == ["git", "tag"]:
            return "dataset-v1.1\n"
        raise AssertionError(cmd)

    monkeypatch.setattr(dvc_tools, "_run", fake_run)
    res = execute_tool("dvc_get_dataset_version", {"path": "data/dataset.csv"}, request_id="d1")
    assert res.status == "ok"
    assert res.data["md5"] == "bbbb" and res.data["git_commit"] == "a4c92e1deadbeef"
    assert res.data["dataset_version"] == "dataset-v1.1"


def test_version_accepts_windows_style_relative_path(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/dataset.csv.dvc").write_text(DVC_B)
    monkeypatch.setenv("DVC_REPO_PATH", str(tmp_path))
    monkeypatch.setattr(dvc_tools, "_run", lambda cmd: {
        "git show": DVC_B,
        "git log": "a4c92e1deadbeef\x1f2026-10-01T10:00:00+00:00\x1fdata: dataset v1.1\n",
        "git tag": "dataset-v1.1\n",
    }[" ".join(cmd[:2])])

    res = execute_tool("dvc_get_dataset_version", {"path": "data\\dataset.csv"}, request_id="d6")

    assert res.status == "ok"
    assert res.data["dvc_file"] == "data/dataset.csv.dvc"
    assert res.data["dataset_version"] == "dataset-v1.1"


def test_list_datasets_returns_posix_paths(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "dataset.csv.dvc").write_text(DVC_B)
    monkeypatch.setenv("DVC_REPO_PATH", str(tmp_path))

    result = dvc_tools.dvc_list_datasets()

    assert result["datasets"][0]["dvc_file"] == "data/dataset.csv.dvc"


def test_compare_versions_detects_a_change(monkeypatch):
    monkeypatch.setattr(dvc_tools, "_run",
                        lambda cmd: DVC_A if "dataset-v1.0:" in cmd[2] else DVC_B)
    res = execute_tool("dvc_compare_versions", {"path": "data/dataset.csv",
                       "rev_a": "dataset-v1.0", "rev_b": "dataset-v1.1"}, request_id="d2")
    assert res.data["identical"] is False and res.data["size_delta_bytes"] == 60


def test_shell_metacharacters_in_revisions_are_rejected():
    res = execute_tool("dvc_compare_versions", {"path": "data/dataset.csv",
                       "rev_a": "main; rm -rf /", "rev_b": "main"}, request_id="d3")
    assert res.status == "error" and res.error_code == "invalid_arguments"


def test_detect_changes_ignores_model_outputs_and_reports_dvc_dataset_changes(
        monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "dataset.csv.dvc").write_text(DVC_B)
    monkeypatch.setenv("DVC_REPO_PATH", str(tmp_path))
    monkeypatch.setattr(
        dvc_tools, "_run",
        lambda cmd: json.dumps({"uncommitted": {
            "modified": ["data/dataset.csv"],
            "added": ["models/model.joblib"],
        }}),
    )

    res = execute_tool("dvc_detect_changes", {}, request_id="d5")

    assert res.status == "ok"
    assert res.data["changed"] is True
    assert res.data["changed_paths"] == ["data/dataset.csv"]


def test_checkout_needs_confirmation(monkeypatch):
    monkeypatch.setattr(dvc_tools, "_run",
                        lambda cmd: (_ for _ in ()).throw(AssertionError("ran")))
    res = execute_tool("dvc_checkout_version", {"path": "data/dataset.csv", "rev": "dataset-v1.0"},
                       request_id="d4")
    assert res.status == "confirmation_required"
