#!/usr/bin/env bash
# Deliberately inject failures, then ask the agent to diagnose them.
#   ./scripts/failures.sh k8s-bad-image | k8s-bad-model-path | dag-corrupt-data | restore
set -euo pipefail
cd "$(dirname "$0")/.."
NS=mlops-agent
case "${1:-}" in
  k8s-bad-image)       # new pods cannot pull the image -> ErrImagePull/ImagePullBackOff
    kubectl -n $NS set image deployment/inference-api inference-api=mlops-inference:does-not-exist ;;
  k8s-bad-model-path)  # pods Running but NOT Ready (readiness 503) -> Service has no endpoints
    kubectl -n $NS patch configmap inference-config --type merge \
      -p '{"data":{"MODEL_PATH":"/models/missing.joblib"}}'
    kubectl -n $NS rollout restart deployment/inference-api ;;
  dag-corrupt-data)    # dataset no longer matches its DVC hash -> validate_dataset fails
    echo "corrupt,row" >> data/dataset.csv
    echo "Now trigger the DAG; also ask the agent whether DVC-tracked data changed." ;;
  restore)
    kubectl -n $NS set image deployment/inference-api inference-api=mlops-inference:1.0.0 || true
    kubectl -n $NS apply -f kubernetes/configmap.yaml
    kubectl -n $NS rollout restart deployment/inference-api || true
    dvc checkout data/dataset.csv.dvc ;;
  *) echo "usage: $0 k8s-bad-image|k8s-bad-model-path|dag-corrupt-data|restore"; exit 2 ;;
esac
