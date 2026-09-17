#!/bin/bash
# Stream F: STAD survival cohort in parallel with stream C (resumable)
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/run_surv.py --grid all --cancers STAD --seeds 0 1 2 3 4
echo "STREAM F DONE"
