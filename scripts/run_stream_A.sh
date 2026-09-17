#!/bin/bash
# Stream A: main protocol (Adam, as in the legacy code) + centralised + sweeps. Resumable.
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
S="0 1 2 3 4"
python scripts/run_cls.py --grid main    --seeds $S --save-ckpt
python scripts/run_cls.py --grid central --seeds $S --save-ckpt
python scripts/run_cls.py --grid mu      --seeds $S
python scripts/run_cls.py --grid lr      --seeds $S
python scripts/run_cls.py --grid participation --fms UNI_v2 Conch_v15 Phikon_v2 Virchow2 --seeds $S
python scripts/run_cls.py --grid linear  --seeds $S
python scripts/run_cls.py --grid controls --seeds $S
python scripts/run_cls.py --grid partition_fedbn --fms UNI_v2 Conch_v15 Phikon_v2 Virchow2 --seeds $S
echo "STREAM A DONE"
