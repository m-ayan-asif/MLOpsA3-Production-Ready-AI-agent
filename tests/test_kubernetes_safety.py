"""Unauthorized Kubernetes modifications must be rejected; reads are namespace-scoped."""
from types import SimpleNamespace

from agent.tools import kubernetes_tools as k8s
from agent.tools.base import execute_tool


class FakeApps:
    def __init__(self):
        self.patches = []

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append((name, ns, body))


def test_restart_in_unapproved_namespace_is_denied_even_with_confirmation(monkeypatch):
    apps = FakeApps()
    monkeypatch.setattr(k8s, "_apps", lambda: apps)
    res = execute_tool("k8s_restart_deployment",
                       {"deployment": "coredns", "namespace": "kube-system"},
                       request_id="k1", confirmed=True)
    assert res.status == "denied" and apps.patches == []


def test_restart_without_confirmation_changes_nothing(monkeypatch):
    apps = FakeApps()
    monkeypatch.setattr(k8s, "_apps", lambda: apps)
    res = execute_tool("k8s_restart_deployment", {"deployment": "inference-api"},
                       request_id="k2", confirmed=False)
    assert res.status == "confirmation_required" and apps.patches == []


def test_restart_with_confirmation_in_approved_namespace_patches(monkeypatch):
    apps = FakeApps()
    monkeypatch.setattr(k8s, "_apps", lambda: apps)
    res = execute_tool("k8s_restart_deployment", {"deployment": "inference-api"},
                       request_id="k3", confirmed=True)
    assert res.status == "ok"
    assert apps.patches[0][:2] == ("inference-api", "mlops-agent")


def test_reads_outside_approved_namespace_are_denied(monkeypatch):
    monkeypatch.setattr(k8s, "_core", lambda: (_ for _ in ()).throw(AssertionError("no API call")))
    res = execute_tool("k8s_get_pod_logs", {"pod_name": "x", "namespace": "kube-system"},
                       request_id="k4")
    assert res.status == "denied"


def test_deployed_version_uses_first_approved_namespace_when_omitted(monkeypatch):
    seen = []
    apps = SimpleNamespace(list_namespaced_deployment=lambda namespace, **kw: (
        seen.append(("deployment", namespace)) or SimpleNamespace(items=[])))
    core = SimpleNamespace(read_namespaced_config_map=lambda name, namespace, **kw: (
        seen.append(("configmap", namespace)) or SimpleNamespace(data={})))
    monkeypatch.setattr(k8s, "_apps", lambda: apps)
    monkeypatch.setattr(k8s, "_core", lambda: core)

    res = execute_tool("k8s_get_deployed_version", {}, request_id="k6")

    assert res.status == "ok"
    assert seen == [("deployment", "mlops-agent"), ("configmap", "mlops-agent")]


def test_deployed_version_rejects_explicit_unapproved_default_namespace(monkeypatch):
    monkeypatch.setattr(k8s, "_apps", lambda: (_ for _ in ()).throw(
        AssertionError("no API call should be made")))
    res = execute_tool("k8s_get_deployed_version", {"namespace": "default"}, request_id="k7")
    assert res.status == "denied"


def test_unhealthy_pod_detection_reports_reason_and_events(monkeypatch):
    def pod(name, waiting=None, ready=True, restarts=0, phase="Running"):
        state = SimpleNamespace(waiting=SimpleNamespace(reason=waiting) if waiting else None)
        cs = SimpleNamespace(ready=ready, restart_count=restarts, state=state, last_state=None,
                             image="mlops-inference:bad")
        return SimpleNamespace(metadata=SimpleNamespace(name=name),
                               status=SimpleNamespace(phase=phase, container_statuses=[cs]),
                               spec=SimpleNamespace(node_name="n1"))

    event = SimpleNamespace(type="Warning", reason="Failed", message="ErrImagePull", count=3,
                            last_timestamp=None, event_time=None)
    core = SimpleNamespace(
        list_namespaced_pod=lambda *a, **kw: SimpleNamespace(
            items=[pod("good"), pod("bad", waiting="ImagePullBackOff", ready=False, phase="Pending")]),
        list_namespaced_event=lambda *a, **kw: SimpleNamespace(items=[event]))
    monkeypatch.setattr(k8s, "_core", lambda: core)
    res = execute_tool("k8s_find_unhealthy_pods", {}, request_id="k5")
    bad = res.data["unhealthy"]
    assert [p["name"] for p in bad] == ["bad"]
    assert bad[0]["waiting_reasons"] == ["ImagePullBackOff"]
    assert bad[0]["warning_events"][0]["message"] == "ErrImagePull"
