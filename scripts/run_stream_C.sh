#!/bin/bash
# Stream C: within-cancer survival. Resumable.
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/run_surv.py --grid all --seeds 0 1 2 3 4
echo "STREAM C DONE"
