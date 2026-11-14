"""DVC data-management tools. Subprocess use is restricted to an allowlist of commands."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agent.config import get_settings
from agent.tools.base import ToolError, tool
from agent.tools.mlflow_tools import _client, resolve_run

ALLOWED_COMMANDS: list[tuple[str, ...]] = [
    ("dvc", "status"),
    ("dvc", "data", "status"),
    ("dvc", "pull"),
    ("git", "rev-parse"),
    ("git", "log"),
    ("git", "show"),
    ("git", "tag"),
    ("git", "checkout"),
]
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^-]*$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]*$")
PROVENANCE_TAGS = ("dvc.path", "dvc.md5", "git.commit", "dataset.version")


def _repo() -> Path:
    return Path(get_settings().dvc_repo_path).resolve()


def _run(cmd: list[str]) -> str:
    if not any(tuple(cmd[: len(p)]) == p for p in ALLOWED_COMMANDS):
        raise ToolError("denied", f"Command not on the DVC/Git allowlist: {' '.join(cmd[:3])}")
    if cmd[:2] == ["git", "checkout"] and (len(cmd) != 5 or cmd[3] != "--"):
        raise ToolError("denied", "Only 'git checkout <rev> -- <file>' is permitted")
    try:
        proc = subprocess.run(cmd, cwd=_repo(), capture_output=True, text=True, timeout=30,
                              check=False, shell=False)
    except FileNotFoundError as exc:
        raise ToolError("dependency_missing", f"'{cmd[0]}' is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError("timeout", f"'{cmd[0]}' timed out") from exc
    if proc.returncode != 0:
        raise ToolError("command_failed", (proc.stderr or proc.stdout).strip()[:400])
    return proc.stdout


def _check_rev(rev: str) -> str:
    if not _REV_RE.match(rev) or ".." in rev:
        raise ToolError("invalid_arguments", f"Invalid git revision: {rev!r}")
    return rev


def _dvc_file(path: str) -> str:
    if not _PATH_RE.match(path) or ".." in path:
        raise ToolError("invalid_arguments", f"Invalid path: {path!r}")
    return path if path.endswith(".dvc") else path + ".dvc"


def _parse_dvc(text: str) -> dict[str, Any]:
    doc = yaml.safe_load(text) or {}
    out = (doc.get("outs") or [{}])[0]
    return {"path": out.get("path"), "md5": out.get("md5"), "size": out.get("size"),
            "nfiles": out.get("nfiles")}


def _read_dvc(dvc_file: str, rev: str | None = None) -> dict[str, Any]:
    if rev:
        return _parse_dvc(_run(["git", "show", f"{_check_rev(rev)}:{dvc_file}"]))
    target = _repo() / dvc_file
    if not target.is_file():
        raise ToolError("not_found", f"{dvc_file} not found in the repository")
    return _parse_dvc(target.read_text())


@tool(source="dvc", description="List datasets tracked by DVC (.dvc files) with hash and size.")
def dvc_list_datasets() -> dict[str, Any]:
    skip = {".git", ".dvc", ".venv", "node_modules"}
    found = []
    for file in sorted(_repo().rglob("*.dvc")):
        rel = file.relative_to(_repo())
        if skip & set(rel.parts):
            continue
        found.append({"dvc_file": str(rel), **_parse_dvc(file.read_text())})
    return {"datasets": found, "count": len(found)}


@tool(
    source="dvc",
    description="Get the current version of one DVC-tracked dataset: content hash, last Git "
                "commit that changed it and any dataset-* Git tag (the version label).",
    properties={"path": {"type": "string", "description": "e.g. data/dataset.csv"}},
    required=("path",),
)
def dvc_get_dataset_version(path: str) -> dict[str, Any]:
    dvc_file = _dvc_file(path)
    info = _read_dvc(dvc_file)
    line = _run(["git", "log", "-1", "--format=%H%x1f%cI%x1f%s", "--", dvc_file]).strip()
    commit = committed_at = message = None
    if line:
        commit, committed_at, message = line.split("\x1f")
    tags = _run(["git", "tag", "--points-at", commit]).split() if commit else []
    version = next((t for t in tags if t.startswith("dataset-")), tags[0] if tags else None)
    return {"dvc_file": dvc_file, **info, "dataset_version": version, "git_commit": commit,
            "committed_at": committed_at, "commit_message": message, "tags": tags}


@tool(
    source="dvc",
    description="Compare a dataset between two Git revisions (tags, branches or commits).",
    properties={"path": {"type": "string"}, "rev_a": {"type": "string"},
                "rev_b": {"type": "string"}},
    required=("path", "rev_a", "rev_b"),
)
def dvc_compare_versions(path: str, rev_a: str, rev_b: str) -> dict[str, Any]:
    dvc_file = _dvc_file(path)
    a, b = _read_dvc(dvc_file, rev_a), _read_dvc(dvc_file, rev_b)
    return {"dvc_file": dvc_file, rev_a: a, rev_b: b, "identical": a["md5"] == b["md5"],
            "size_delta_bytes": (b["size"] or 0) - (a["size"] or 0)}


@tool(source="dvc",
      description="Detect whether DVC-tracked data in the workspace differs from what DVC recorded.")
def dvc_detect_changes() -> dict[str, Any]:
    raw = json.loads(_run(["dvc", "data", "status", "--json"]) or "{}")
    uncommitted = raw.get("uncommitted", {}) or {}
    changed = [*uncommitted.get("modified", []), *uncommitted.get("deleted", [])]
    return {"changed": bool(changed), "changed_paths": changed,
            "not_in_cache": raw.get("not_in_cache", []), "raw": raw}


@tool(
    source="dvc",
    description="Explain, step by step, how to reproduce a dataset at a given Git revision. "
                "Read-only: it verifies the revision exists and returns commands.",
    properties={"path": {"type": "string"}, "rev": {"type": "string"}},
    required=("path", "rev"),
)
def dvc_explain_reproduce(path: str, rev: str) -> dict[str, Any]:
    dvc_file = _dvc_file(path)
    info = _read_dvc(dvc_file, rev)  # fails if the revision or file does not exist
    return {"target_md5": info["md5"], "steps": [
        f"git checkout {rev} -- {dvc_file}   # restore the recorded hash",
        f"dvc pull {dvc_file}                 # fetch that exact content from the DVC remote",
        "dvc data status                     # confirm workspace matches (no modified files)",
    ]}


@tool(
    source="dvc",
    description="Return the dataset provenance recorded on an MLflow run (DVC hash, Git commit, "
                "dataset version). Accepts a run id, a run name or a numeric experiment id "
                "(uses that experiment's latest run).",
    properties={"reference": {"type": "string"}},
    required=("reference",),
)
def dvc_get_run_dataset(reference: str) -> dict[str, Any]:
    run, resolved_as = resolve_run(_client(), reference)
    tags = run.data.tags
    prov = {k: tags.get(k) for k in PROVENANCE_TAGS}
    if not prov["dvc.md5"]:
        raise ToolError("no_provenance", f"Run {run.info.run_id} has no recorded DVC provenance")
    return {"run_id": run.info.run_id, "resolved_as": resolved_as, **prov}


@tool(
    source="dvc",
    mutating=True,
    description="Check out a dataset at a Git revision (git checkout <rev> -- file.dvc, then "
                "dvc pull). Overwrites the workspace copy of the dataset.",
    properties={"path": {"type": "string"}, "rev": {"type": "string"}},
    required=("path", "rev"),
)
def dvc_checkout_version(path: str, rev: str, confirm: bool = False) -> dict[str, Any]:
    dvc_file = _dvc_file(path)
    _check_rev(rev)
    if not confirm:
        raise ToolError("confirmation_required",
                        f"This would overwrite the workspace copy of {dvc_file} with revision "
                        f"{rev}. Ask the user to confirm.")
    _run(["git", "checkout", rev, "--", dvc_file])
    _run(["dvc", "pull", dvc_file])
    return {"checked_out": dvc_file, "rev": rev, "md5": _read_dvc(dvc_file)["md5"]}
