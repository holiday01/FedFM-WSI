#!/usr/bin/env python
"""Within-cancer survival grid. Resumable; see fedfm/survival.py."""
import os
import sys
import json
import time
import hashlib
import argparse
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "code"))

import numpy as np
import torch

from fedfm import data as D
from fedfm.survival import SurvConfig, run_survival

OUT = REV / "results" / "surv"
LRS = [1e-4, 3e-4, 1e-3]


def configs(fm, cancer, seeds, grid):
    C = []
    for s in seeds:
        if grid in ("main", "all"):
            for lr in LRS:
                for mode in ("fl", "central", "central_strat"):
                    C.append(SurvConfig(fm=fm, cancer=cancer, mode=mode, lr=lr, seed=s))
        if grid in ("legacy", "all"):
            C.append(SurvConfig(fm=fm, cancer=cancer, mode="fl_legacy_restore", seed=s))
            C.append(SurvConfig(fm=fm, cancer=cancer, mode="central_legacy", seed=s))
    return C


def key_of(cfg):
    return hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]


def cohort(cancer):
    p = REV / "data" / f"cohort_survival_{cancer}.json"
    if p.exists():
        return json.load(open(p))
    s = D.build_survival_cohort(f"TCGA-{cancer}")
    for x in s:
        x["age"] = None if x["age"] != x["age"] else float(x["age"])
        x["stage"] = None if not isinstance(x["stage"], str) else x["stage"]
    json.dump(s, open(p, "w"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fms", nargs="+", default=D.FMS)
    ap.add_argument("--cancers", nargs="+", default=["BRCA", "COAD", "STAD"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--grid", default="all")
    args = ap.parse_args()
    dev = "cuda"
    for cancer in args.cancers:
        samples = cohort(cancer)
        fn = [s["filename"] for s in samples]
        for fm in args.fms:
            cfgs = [c for c in configs(fm, cancer, args.seeds, args.grid)
                    if not (OUT / cancer / fm / f"{key_of(c)}.json").exists()]
            print(f"[surv] {cancer} {fm}: {len(cfgs)} to run", flush=True)
            if not cfgs:
                continue
            X = torch.tensor(np.asarray(D.load_feature_matrix(fm, fn)), device=dev)
            (OUT / cancer / fm).mkdir(parents=True, exist_ok=True)
            for cfg in cfgs:
                k = key_of(cfg)
                rec, pred = run_survival(cfg, X, samples, dev)
                rec["key"] = k
                json.dump(rec, open(OUT / cancer / fm / f"{k}.json", "w"))
                np.savez_compressed(OUT / cancer / fm / f"{k}_pred.npz", **pred)
                print(f"  {cancer} {fm} {cfg.mode} lr={cfg.lr} seed={cfg.seed} "
                      f"val={rec['best_val']:.3f} C_pat={rec['test_c_patient']:.3f} "
                      f"best@{rec['best_round']} ({rec['seconds']:.0f}s)", flush=True)
            del X
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
