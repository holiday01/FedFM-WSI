#!/bin/bash
# Stream B: SGD optimiser control, partitions, selection. Resumable.
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
S="0 1 2 3 4"
python scripts/run_cls.py --grid sgd       --seeds $S --save-ckpt
python scripts/run_cls.py --grid partition --seeds $S
python scripts/run_cls.py --grid partition_prox --seeds $S
python scripts/run_cls.py --grid selection --fms UNI_v2 --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_cls.py --grid selection --fms Conch_v15 Phikon_v2 --seeds $S
python scripts/run_cls.py --grid sgd_mu    --fms UNI_v2 Conch_v15 Phikon_v2 --seeds $S
echo "STREAM B DONE"
