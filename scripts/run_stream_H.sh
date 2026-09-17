#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/run_cls.py --grid partition_fedbn_sgd --fms UNI_v2 Conch_v15 Phikon_v2 Virchow2 --seeds 0 1 2 3 4
python scripts/run_cls.py --grid linear2 --seeds 0 1 2 3 4
echo "STREAM H DONE"
