#!/usr/bin/env python
"""
Run classification experiment grids.

  python scripts/run_cls.py --grid main --fms UNI_v2 Conch_v15 --seeds 0 1 2 3 4

Every run is keyed by a hash of its full configuration; finished runs are skipped, so the
script is resumable.  Outputs: results/cls/<grid>/<fm>/<key>.json (+ _pred.npz, + .pt)
"""
import os
import sys
import json
import hashlib
import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "code"))

import numpy as np
import torch

from fedfm import data as D
from fedfm.fl import FLConfig, FLRun
from fedfm.central import CentralConfig, run_central
from fedfm.partitions import client_assignment, partition_summary
from fedfm.selectors import make_selector

OUT = REV / "results" / "cls"
SGD_LRS = [0.003, 0.01, 0.03, 0.1]


def grid_configs(grid, fm, seeds):
    C = []
    if grid == "main":
        for s in seeds:
            for algo in ["FedAvg", "FedProx", "SCAFFOLD", "FedBN"]:
                C.append(FLConfig(fm=fm, algorithm=algo, seed=s))
    elif grid == "lr":
        for s in seeds:
            for lr in [1e-4, 1e-3]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", lr=lr, seed=s))
    elif grid == "mu":
        for s in seeds:
            for mu in [0.001, 0.1, 1.0]:
                C.append(FLConfig(fm=fm, algorithm="FedProx", mu=mu, seed=s))
    elif grid == "central":                   # tuned protocol: Adam and SGD, FL-style early stopping
        for s in seeds:
            for lr in [1e-4, 3e-4, 1e-3]:
                C.append(CentralConfig(fm=fm, protocol="tuned", lr=lr, seed=s))
            for lr in SGD_LRS:
                C.append(CentralConfig(fm=fm, protocol="tuned", optimizer="sgd", lr=lr, seed=s))
            C.append(CentralConfig(fm=fm, protocol="legacy", seed=s))
    elif grid == "lr_all":                    # Adam learning rate selected per algorithm (FedAvg is in `lr`)
        for s in seeds:
            for algo in ["FedProx", "SCAFFOLD", "FedBN"]:
                for lr in [1e-4, 1e-3]:
                    C.append(FLConfig(fm=fm, algorithm=algo, lr=lr, seed=s))
    elif grid == "partition_prox2":           # FedProx on the remaining partitions
        for s in seeds:
            for p in ["project_tss", "labelskew_0.1", "iid"]:
                C.append(FLConfig(fm=fm, algorithm="FedProx", partition=p, sampling="uniform",
                                  aggregation="sample_weighted", seed=s))
    elif grid == "partition_bnref":           # aggregated-BN reference (same BN architecture as FedBN)
        for s in seeds:
            C.append(FLConfig(fm=fm, algorithm="FedAvgBN", partition="institution_all", sampling="uniform",
                              aggregation="sample_weighted", seed=s))
            for lr in [0.03, 0.1]:
                C.append(FLConfig(fm=fm, algorithm="FedAvgBN", optimizer="sgd", lr=lr, partition="institution_all",
                                  sampling="uniform", aggregation="sample_weighted", seed=s))
    elif grid == "sgd":                       # optimiser control, all algorithms
        for s in seeds:
            for algo in ["FedAvg", "FedProx", "SCAFFOLD", "FedBN"]:
                for lr in SGD_LRS:
                    C.append(FLConfig(fm=fm, algorithm=algo, optimizer="sgd", lr=lr, seed=s))
    elif grid == "sgd_mu":                    # FedProx mu sweep under SGD (validation-selected)
        for s in seeds:
            for lr in SGD_LRS:
                for mu in [0.001, 0.1, 1.0]:
                    C.append(FLConfig(fm=fm, algorithm="FedProx", optimizer="sgd", lr=lr, mu=mu, seed=s))
    elif grid == "participation":
        for s in seeds:
            for k in [9, 15, 30, None]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", clients_per_round=k, seed=s))
    elif grid == "partition":
        for s in seeds:
            for p in ["project_tss", "institution", "labelskew_0.1", "labelskew_1.0", "iid"]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", partition=p, sampling="uniform",
                                  aggregation="sample_weighted", seed=s))
    elif grid == "partition_prox":
        for s in seeds:
            for p in ["institution", "labelskew_1.0"]:
                C.append(FLConfig(fm=fm, algorithm="FedProx", partition=p, sampling="uniform",
                                  aggregation="sample_weighted", seed=s))
    elif grid == "partition_fedbn":           # FedBN on multi-class (institution) clients
        for s in seeds:
            for algo in ["FedAvg", "FedBN"]:
                C.append(FLConfig(fm=fm, algorithm=algo, partition="institution_all", sampling="uniform",
                                  aggregation="sample_weighted", seed=s))
    elif grid == "controls":
        for s in seeds:
            for ft in ["zscore", "zscore_rp"]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", feature_transform=ft, seed=s))
                C.append(CentralConfig(fm=fm, protocol="tuned", lr=3e-4, feature_transform=ft, seed=s))
    elif grid == "controls2":                 # standardised features x optimiser / algorithm (follow-up)
        for s in seeds:
            for lr in [0.01, 0.03]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", optimizer="sgd", lr=lr, feature_transform="zscore", seed=s))
            for algo in ["FedProx", "SCAFFOLD"]:
                C.append(FLConfig(fm=fm, algorithm=algo, feature_transform="zscore", seed=s))
    elif grid == "linear2":                   # linear probe with standardised features (Adam and SGD)
        for s in seeds:
            C.append(FLConfig(fm=fm, algorithm="FedAvg", arch="linear", feature_transform="zscore", seed=s))
            for lr in [0.01, 0.03]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", arch="linear", optimizer="sgd", lr=lr, feature_transform="zscore", seed=s))
            C.append(CentralConfig(fm=fm, protocol="tuned", arch="linear", lr=1e-4, feature_transform="zscore", seed=s))
    elif grid == "partition_fedbn_sgd":       # FedBN / FedAvg on institution clients with SGD
        for s in seeds:
            for algo in ["FedAvg", "FedBN"]:
                for lr in [0.03, 0.1]:
                    C.append(FLConfig(fm=fm, algorithm=algo, optimizer="sgd", lr=lr, partition="institution_all",
                                      sampling="uniform", aggregation="sample_weighted", seed=s))
    elif grid == "linear":
        for s in seeds:
            C.append(FLConfig(fm=fm, algorithm="FedAvg", arch="linear", seed=s))
            for lr in [1e-4, 3e-4, 1e-3]:
                C.append(CentralConfig(fm=fm, protocol="tuned", arch="linear", lr=lr, seed=s))
    elif grid == "legacy_droplast":
        for s in seeds:
            C.append(FLConfig(fm=fm, algorithm="FedAvg", legacy_drop_last=True, seed=s))
            C.append(FLConfig(fm=fm, algorithm="SCAFFOLD", legacy_drop_last=True, seed=s))
    elif grid == "selection":                 # stratified / uniform random / UCB1 / PathologyAware (v2 reward)
        for s in seeds:
            for sel in ["stratified", "uniform", "ucb", "pathology_aware"]:
                C.append(FLConfig(fm=fm, algorithm="FedAvg", sampling=sel, seed=s))
    elif grid == "pilot_sgd":
        for s in seeds:
            for algo in ["FedAvg", "SCAFFOLD"]:
                for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
                    C.append(FLConfig(fm=fm, algorithm=algo, optimizer="sgd", lr=lr, seed=s))
    else:
        raise ValueError(grid)
    return C


def key_of(cfg):
    d = asdict(cfg)
    d["_type"] = type(cfg).__name__
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--fms", nargs="+", default=D.FMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    dev = "cuda"

    coh_path = REV / "data" / "cohort_classification.json"
    if coh_path.exists():
        samples = json.load(open(coh_path))
    else:
        samples = D.build_classification_cohort()
        json.dump(samples, open(coh_path, "w"))
    fnames = [s["filename"] for s in samples]
    y = torch.tensor([s["label"] for s in samples], dtype=torch.long, device=dev)

    for fm in args.fms:
        cfgs = grid_configs(args.grid, fm, args.seeds)
        todo = [c for c in cfgs if not (OUT / args.grid / fm / f"{key_of(c)}.json").exists()]
        print(f"[{args.grid}] {fm}: {len(cfgs)} configs, {len(todo)} to run", flush=True)
        if not todo:
            continue
        X = torch.tensor(np.asarray(D.load_feature_matrix(fm, fnames)), device=dev)
        (OUT / args.grid / fm).mkdir(parents=True, exist_ok=True)
        for cfg in todo:
            k = key_of(cfg)
            t0 = time.time()
            if isinstance(cfg, CentralConfig):
                rec, pred, state = run_central(cfg, X, y, samples, dev)
            else:
                client_of = client_assignment(samples, cfg.partition)
                run = FLRun(cfg, X, y, samples, client_of, dev, verbose=args.verbose)
                if cfg.sampling in ("ucb", "pathology_aware"):
                    run.selector = make_selector(cfg.sampling, run.clients, X)
                rec, pred, state = run.run()
                rec["partition_summary"] = partition_summary(samples, client_of)
                rec["n_clients"] = len(run.clients)
            rec["key"] = k
            rec["grid"] = args.grid
            json.dump(rec, open(OUT / args.grid / fm / f"{k}.json", "w"))
            np.savez_compressed(OUT / args.grid / fm / f"{k}_pred.npz", **pred)
            if args.save_ckpt:
                torch.save({kk: v.half().cpu() if v.is_floating_point() else v.cpu()
                            for kk, v in state.items()}, OUT / args.grid / fm / f"{k}.pt")
            t = rec["test"]
            extra = f"rounds={rec.get('rounds_run')} best@{rec.get('best_round', rec.get('best_epoch'))}"
            name = getattr(cfg, "algorithm", "central-" + getattr(cfg, "protocol", ""))
            print(f"  {fm} {name} seed={cfg.seed} {cfg.tag} val={rec['best_val']:.4f} "
                  f"test_acc={t['accuracy']:.4f} bacc={t['balanced_accuracy']:.4f} {extra} "
                  f"({time.time()-t0:.0f}s) [{k}]", flush=True)
        del X
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
