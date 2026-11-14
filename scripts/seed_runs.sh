#!/usr/bin/env bash
# Creates several MLflow runs with different hyperparameters.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
python -m training.train --n-estimators 10  --max-depth 2 --run-name rf-n10-d2
python -m training.train --n-estimators 50  --max-depth 4 --run-name rf-n50-d4
python -m training.train --n-estimators 100 --max-depth 6 --run-name rf-n100-d6
python -m training.train --n-estimators 200 --max-depth 8 --run-name rf-n200-d8
python -m training.train --n-estimators 300 --run-name rf-n300-unbounded
