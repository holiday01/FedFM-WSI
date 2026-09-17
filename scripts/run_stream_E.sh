#!/bin/bash
# Stream E: legacy-batching reproduction, dimensionality controls, linear probe (resumable; overlaps with A/D are skipped by hash)
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
S="0 1 2 3 4"
python scripts/run_cls.py --grid legacy_droplast --seeds $S
python scripts/run_cls.py --grid controls --seeds $S
python scripts/run_cls.py --grid linear --seeds $S
echo "STREAM E DONE"
