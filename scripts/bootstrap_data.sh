#!/usr/bin/env bash
# Creates two DVC-tracked dataset versions (dataset-v1.0, dataset-v1.1) with a local remote.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .git ] || git init -q -b main
[ -d .dvc ] || dvc init -q
dvc remote list | grep -q local || dvc remote add -d local "$(pwd)/../mlops-dvc-remote"

python -m training.make_data --version v1
dvc add data/dataset.csv
git add data/dataset.csv.dvc data/.gitignore .dvc dvc.yaml .gitignore
git commit -q -m "data: dataset v1.0 (569 rows)"
git tag -f dataset-v1.0

python -m training.make_data --version v2
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -q -m "data: dataset v1.1 (+300 augmented rows)"
git tag -f dataset-v1.1
dvc push
echo "OK: tags dataset-v1.0 and dataset-v1.1 created and pushed to the DVC remote"
