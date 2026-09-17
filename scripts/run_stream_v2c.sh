#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4; S="0 1 2 3 4"
python scripts/run_cls.py --grid lr_all --seeds $S
echo "STREAM V2C DONE"
