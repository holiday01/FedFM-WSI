#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/run_cls.py --grid controls2 --seeds 0 1 2 3 4
echo "STREAM G DONE"
