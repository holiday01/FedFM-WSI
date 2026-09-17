#!/bin/bash
# Stream D: waits for stream A, then reproduces the legacy drop_last behaviour.
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
until grep -q "STREAM A DONE" logs/stream_A.log; do sleep 60; done
python scripts/run_cls.py --grid legacy_droplast --seeds 0 1 2 3 4
echo "STREAM D DONE"
