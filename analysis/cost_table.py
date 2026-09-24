#!/usr/bin/env python
"""
End-to-end cost per encoder (R3 minor 1): feature-extraction throughput (from the TRIDENT
logs of the CPTAC run, same GPU and pipeline as TCGA), slide-level cache size, trainable
head size, bytes per round and bytes to the best round (from the main FedAvg runs).
Output analysis/tables/T_cost.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

REV = Path(__file__).resolve().parents[1]
T = REV / "analysis" / "tables"
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
TILE = {"UNI_v2": 1536, "Virchow2": 2560, "Phikon_v2": 1024, "Conch_v15": 768, "CTransPath": 768, "Midnight12k": 1536, "ResNet50": 1024}
PARAMS_M = {"UNI_v2": 682, "Virchow2": 632, "Phikon_v2": 307, "Conch_v15": 304, "CTransPath": 28, "Midnight12k": 1136, "ResNet50": 25.6}
TR = {"UNI_v2": "uni_v2", "Virchow2": "virchow2", "Phikon_v2": "phikon_v2", "Conch_v15": "conch_v15",
      "CTransPath": "ctranspath", "Midnight12k": "midnight12k", "ResNet50": "resnet50"}

ext = pd.read_csv(T / "T_extraction_time_raw.csv")
man = pd.read_csv(REV / "data" / "external" / "manifest.csv") if (REV / "data" / "external" / "manifest.csv").exists() else None
n_slides = 5208
rows = []
for fm in FMS:
    e = ext[(ext.fm == TR[fm]) & (ext.slides_done > 0)]
    # seconds per 1,000 tiles: elapsed time of every logged section divided by the number of
    # slides it processed times the mean tile count of that cohort (from the manifest)
    sec_per_ktile = np.nan
    if man is not None and len(e):
        mt = man[man.n_tiles > 0].groupby("cohort").n_tiles.mean()
        tot_s = float(e.elapsed_s.sum())
        tot_t = float(sum(r.slides_done * mt.get(r.cohort, np.nan) for _, r in e.iterrows()))
        if tot_t > 0:
            sec_per_ktile = 1000 * tot_s / tot_t
    cache_bytes = n_slides * 8 * TILE[fm] * 8                       # float64 cache as stored
    rows.append(dict(fm=fm, encoder_params_M=PARAMS_M[fm], tile_dim=TILE[fm], slide_dim=8 * TILE[fm],
                     sec_per_ktile=sec_per_ktile, cache_MB=cache_bytes / 1e6))
df = pd.DataFrame(rows).set_index("fm")
df["cache_float32_MB"] = df.cache_MB / 2
head = pd.read_csv(T / "T_cost_head.csv")
# one row per (fm, setting): MLP FedAvg Adam/SGD, SCAFFOLD (control variates are exchanged in both directions, doubling the traffic), linear head
head = head.merge(df[["encoder_params_M", "tile_dim", "slide_dim", "sec_per_ktile", "cache_MB", "cache_float32_MB"]], left_on="fm", right_index=True)
head["head_MB"] = head.n_params * 4 / 1e6
head["MB_per_round"] = head.bytes_per_round_total / 1e6
head.to_csv(T / "T_cost.csv", index=False)
print(head[["setting", "fm", "n_params", "head_MB", "MB_per_round", "best_round", "rounds_run", "GB_to_best", "GB_to_stop", "seconds"]].round(2).to_string())
# survival head size
surv_params = {fm: (8 * TILE[fm]) * 512 + 512 + 512 * 256 + 256 + 256 + 1 for fm in FMS}
pd.DataFrame(dict(fm=list(surv_params), survival_head_params=list(surv_params.values()))).to_csv(T / "T_cost_surv_head.csv", index=False)
