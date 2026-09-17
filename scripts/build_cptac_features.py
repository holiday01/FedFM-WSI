#!/usr/bin/env python
"""
Build slide-level CPTAC features (external, non-TCGA cohort) with EXACTLY the
aggregation used for the TCGA cache: per-dimension [mean, std(ddof=0), max, min,
p25, median, p75, p90] over tiles, concatenated (8 x D).  Verified against
/mnt/10t/cached_slide_features_npy on TCGA slides (max abs error 0.0).
Output: data/external/<FM>/<slide>.npy (float64, like the TCGA cache) + manifest.
"""
import os, sys, glob, json, time
import numpy as np, h5py, pandas as pd
from pathlib import Path
REV = Path(__file__).resolve().parents[1]
OUT = REV / "data" / "external"
SRC = Path("/mnt/10t/CPTAC")
FM_DIR = {"UNI_v2": "uni_v2", "Virchow2": "virchow2", "Phikon_v2": "phikon_v2", "Conch_v15": "conch_v15",
          "CTransPath": "ctranspath", "Midnight12k": "midnight12k", "ResNet50": "resnet50"}
COHORT = {"luad": ("LUAD", 4), "pda": ("PAAD", 8)}      # label ids as in the TCGA cohort

def agg(T):
    T = T.astype(np.float64)
    return np.concatenate([T.mean(0), T.std(0), T.max(0), T.min(0),
                           np.percentile(T, 25, 0), np.median(T, 0),
                           np.percentile(T, 75, 0), np.percentile(T, 90, 0)])

def main():
    rows = []
    for coh, (proj, lab) in COHORT.items():
        for fm, sub in FM_DIR.items():
            d = SRC / f"trident_{coh}" / "20x_256px_0px_overlap" / f"features_{sub}"
            od = OUT / fm; od.mkdir(parents=True, exist_ok=True)
            files = sorted(glob.glob(str(d / "*.h5")))
            t0 = time.time()
            for f in files:
                sid = Path(f).stem
                o = od / f"{coh}__{sid}.npy"
                if o.exists():
                    rows.append(dict(cohort=coh, project=proj, label=lab, fm=fm, slide=sid, n_tiles=-1)); continue
                with h5py.File(f, "r") as h:
                    T = h["features"][()]
                if T.shape[0] == 0:
                    print("EMPTY", f); continue
                np.save(o, agg(T))
                rows.append(dict(cohort=coh, project=proj, label=lab, fm=fm, slide=sid, n_tiles=int(T.shape[0])))
            print(f"{coh} {fm}: {len(files)} slides, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
