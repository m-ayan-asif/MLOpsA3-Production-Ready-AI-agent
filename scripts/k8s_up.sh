#!/usr/bin/env bash
# Local cluster + deploy. Requires: docker, kind, kubectl. Run after training exported models/model.joblib.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f models/model.joblib ] || { echo "run: python -m training.train --export models/model.joblib"; exit 1; }
kind get clusters | grep -qx mlops || kind create cluster --name mlops
docker build -t mlops-inference:1.0.0 .
kind load docker-image mlops-inference:1.0.0 --name mlops
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl -n mlops-agent rollout status deployment/inference-api --timeout=120s
kubectl -n mlops-agent get pods,svc
