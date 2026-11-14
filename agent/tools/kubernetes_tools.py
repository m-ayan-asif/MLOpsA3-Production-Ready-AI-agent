"""Kubernetes tools. Every call is restricted to approved namespaces; writes need confirmation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import get_settings
from agent.tools.base import ToolError, tool

REQUEST_TIMEOUT = 10
HEALTHY_PHASES = {"Running", "Succeeded"}


def _api_client():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:  # noqa: BLE001 - not running in a cluster
        config.load_kube_config()
    return client


def _core():
    return _api_client().CoreV1Api()


def _apps():
    return _api_client().AppsV1Api()


def _ns(namespace: str | None) -> str:
    allowed = get_settings().k8s_allowed_namespaces
    ns = namespace or allowed[0]
    if ns not in allowed:
        raise ToolError("denied", f"Namespace '{ns}' is not approved (allowed: {list(allowed)})")
    return ns


def _pod_view(pod) -> dict[str, Any]:
    statuses = pod.status.container_statuses or []
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": bool(statuses) and all(cs.ready for cs in statuses),
        "restarts": sum(cs.restart_count for cs in statuses),
        "waiting_reasons": [cs.state.waiting.reason for cs in statuses
                            if cs.state and cs.state.waiting],
        "last_terminated_reasons": [cs.last_state.terminated.reason for cs in statuses
                                    if cs.last_state and cs.last_state.terminated],
        "images": [cs.image for cs in statuses],
        "node": pod.spec.node_name,
    }


def _is_unhealthy(view: dict[str, Any]) -> bool:
    return (not view["ready"] or view["restarts"] > 0 or bool(view["waiting_reasons"])
            or view["phase"] not in HEALTHY_PHASES)


def _events(core, ns: str, pod_name: str) -> list[dict[str, Any]]:
    items = core.list_namespaced_event(
        ns, field_selector=f"involvedObject.name={pod_name}", _request_timeout=REQUEST_TIMEOUT
    ).items
    items.sort(key=lambda e: e.last_timestamp or e.event_time or datetime.min.replace(
        tzinfo=timezone.utc))
    return [{"type": e.type, "reason": e.reason, "message": (e.message or "")[:300],
             "count": e.count} for e in items[-10:]]


@tool(
    source="kubernetes",
    description="List the application's Pods and Deployments (status, readiness, replicas).",
    properties={"namespace": {"type": "string"}},
)
def k8s_list_workloads(namespace: str | None = None) -> dict[str, Any]:
    ns, s = _ns(namespace), get_settings()
    pods = _core().list_namespaced_pod(ns, label_selector=s.k8s_app_label,
                                       _request_timeout=REQUEST_TIMEOUT).items
    deps = _apps().list_namespaced_deployment(ns, label_selector=s.k8s_app_label,
                                              _request_timeout=REQUEST_TIMEOUT).items
    return {"namespace": ns,
            "pods": [_pod_view(p) for p in pods],
            "deployments": [{"name": d.metadata.name, "desired": d.spec.replicas,
                             "ready": d.status.ready_replicas or 0,
                             "updated": d.status.updated_replicas or 0,
                             "available": d.status.available_replicas or 0} for d in deps]}


@tool(
    source="kubernetes",
    description="Describe one Pod: container states, probe config and recent events.",
    properties={"pod_name": {"type": "string"}, "namespace": {"type": "string"}},
    required=("pod_name",),
)
def k8s_describe_pod(pod_name: str, namespace: str | None = None) -> dict[str, Any]:
    ns, core = _ns(namespace), _core()
    pod = core.read_namespaced_pod(pod_name, ns, _request_timeout=REQUEST_TIMEOUT)
    probes = [{"container": c.name,
               "readiness": c.readiness_probe.http_get.path if c.readiness_probe
               and c.readiness_probe.http_get else None,
               "liveness": c.liveness_probe.http_get.path if c.liveness_probe
               and c.liveness_probe.http_get else None} for c in pod.spec.containers]
    return {**_pod_view(pod), "probes": probes, "events": _events(core, ns, pod_name)}


@tool(
    source="kubernetes",
    description="Fetch recent log lines from a Pod (set previous=true for the last crashed container).",
    properties={"pod_name": {"type": "string"}, "namespace": {"type": "string"},
                "tail_lines": {"type": "integer", "minimum": 5, "maximum": 300},
                "previous": {"type": "boolean"}},
    required=("pod_name",),
)
def k8s_get_pod_logs(pod_name: str, namespace: str | None = None, tail_lines: int = 80,
                     previous: bool = False) -> dict[str, Any]:
    ns = _ns(namespace)
    text = _core().read_namespaced_pod_log(pod_name, ns, tail_lines=tail_lines,
                                           previous=previous, _request_timeout=REQUEST_TIMEOUT)
    return {"pod": pod_name, "previous": previous, "log_tail": text.splitlines()[-tail_lines:]}


@tool(
    source="kubernetes",
    description="Inspect Services (type, ports) and whether they have ready endpoints.",
    properties={"namespace": {"type": "string"}},
)
def k8s_list_services(namespace: str | None = None) -> dict[str, Any]:
    ns, core = _ns(namespace), _core()
    out = []
    for svc in core.list_namespaced_service(ns, _request_timeout=REQUEST_TIMEOUT).items:
        try:
            ep = core.read_namespaced_endpoints(svc.metadata.name, ns,
                                                _request_timeout=REQUEST_TIMEOUT)
            ready = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
        except Exception:  # noqa: BLE001
            ready = None
        out.append({"name": svc.metadata.name, "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip, "selector": svc.spec.selector,
                    "ports": [{"port": p.port, "target_port": str(p.target_port)}
                              for p in svc.spec.ports or []], "ready_endpoints": ready})
    return {"namespace": ns, "services": out}


@tool(
    source="kubernetes",
    description="Find unhealthy Pods (not ready, restarts, CrashLoopBackOff, ImagePullBackOff, "
                "OOMKilled) with their recent Warning events.",
    properties={"namespace": {"type": "string"}},
)
def k8s_find_unhealthy_pods(namespace: str | None = None) -> dict[str, Any]:
    ns, s, core = _ns(namespace), get_settings(), _core()
    pods = core.list_namespaced_pod(ns, label_selector=s.k8s_app_label,
                                    _request_timeout=REQUEST_TIMEOUT).items
    bad = []
    for pod in pods:
        view = _pod_view(pod)
        if _is_unhealthy(view):
            warnings = [e for e in _events(core, ns, pod.metadata.name) if e["type"] == "Warning"]
            bad.append({**view, "warning_events": warnings})
    return {"namespace": ns, "total_pods": len(pods), "unhealthy": bad}


@tool(
    source="kubernetes",
    description="Report the currently deployed application version: image tags, version label, "
                "ConfigMap APP_VERSION/MODEL_VERSION and replica readiness.",
    properties={"namespace": {"type": "string"}},
)
def k8s_get_deployed_version(namespace: str | None = None) -> dict[str, Any]:
    ns, s = _ns(namespace), get_settings()
    deps = _apps().list_namespaced_deployment(ns, label_selector=s.k8s_app_label,
                                              _request_timeout=REQUEST_TIMEOUT).items
    try:
        cm = _core().read_namespaced_config_map(s.k8s_configmap, ns,
                                                _request_timeout=REQUEST_TIMEOUT).data or {}
    except Exception:  # noqa: BLE001
        cm = {}
    return {"namespace": ns,
            "configmap": {k: cm.get(k) for k in ("APP_VERSION", "MODEL_VERSION", "MODEL_PATH")},
            "deployments": [{
                "name": d.metadata.name,
                "images": [c.image for c in d.spec.template.spec.containers],
                "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
                "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0} for d in deps]}


@tool(
    source="kubernetes",
    mutating=True,
    description="Rolling-restart a Deployment in an approved namespace.",
    properties={"deployment": {"type": "string"}, "namespace": {"type": "string"}},
    required=("deployment",),
)
def k8s_restart_deployment(deployment: str, namespace: str | None = None,
                           confirm: bool = False) -> dict[str, Any]:
    ns = _ns(namespace)  # namespace policy is checked before anything else
    if not confirm:
        raise ToolError("confirmation_required",
                        f"Restarting deployment '{deployment}' in '{ns}' changes cluster state. "
                        "Ask the user to confirm.")
    stamp = datetime.now(timezone.utc).isoformat()
    body = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": stamp}}}}}
    _apps().patch_namespaced_deployment(deployment, ns, body, _request_timeout=REQUEST_TIMEOUT)
    return {"restarted": deployment, "namespace": ns, "at": stamp}
