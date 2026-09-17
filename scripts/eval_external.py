#!/usr/bin/env python
"""
External (non-TCGA) evaluation of trained heads on CPTAC-LUAD and CPTAC-PDA.

Every saved checkpoint of the `main`, `sgd` and `central` grids (FedBN excluded: it needs
per-client BN state; feature_transform must be raw) is applied, without adaptation, to the
CPTAC slide features built by scripts/build_cptac_features.py.  Reported: recall of the
LUAD (label 4) and PAAD (label 8) classes and the predicted-class distribution.
Output: results/external/external_eval.csv
"""
import os, sys, json, glob
from pathlib import Path
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "code"))
import numpy as np, pandas as pd, torch
from fedfm.models import MLPClassifier, LinearProbe

EXT = REV / "data" / "external"
OUT = REV / "results" / "external"; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]
COH = {"luad": 4, "pda": 8}


def load_ext(fm):
    X, y, coh = [], [], []
    for c, lab in COH.items():
        for f in sorted((EXT / fm).glob(f"{c}__*.npy")):
            X.append(np.load(f).astype(np.float32)); y.append(lab); coh.append(c)
    return np.stack(X), np.array(y), np.array(coh)


def build(cfg, input_dim):
    if cfg.get("arch", "mlp") == "linear":
        return LinearProbe(input_dim, 9)
    return MLPClassifier(input_dim, 9, tuple(cfg.get("hidden_dims", (512, 256))), cfg.get("dropout", 0.3), False)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for fm in ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]:
        if not (EXT / fm).exists():
            continue
        X, y, coh = load_ext(fm)
        Xt = torch.tensor(X, device=dev)
        for grid in ["main", "sgd", "central"]:
            for pt in sorted((REV / "results" / "cls" / grid / fm).glob("*.pt")):
                j = json.load(open(str(pt)[:-3] + ".json")); cfg = j["config"]
                if cfg.get("algorithm") == "FedBN" or cfg.get("feature_transform", "raw") != "raw":
                    continue
                m = build(cfg, X.shape[1]).to(dev)
                st = {k: v.float() if v.is_floating_point() else v for k, v in torch.load(pt, map_location=dev).items()}
                m.load_state_dict(st); m.eval()
                with torch.no_grad():
                    pred = m(Xt).argmax(1).cpu().numpy()
                for c, lab in COH.items():
                    msk = coh == c
                    dist = np.bincount(pred[msk], minlength=9) / msk.sum()
                    rows.append(dict(grid=grid, fm=fm, algorithm=cfg.get("algorithm", "central-" + cfg.get("protocol", "")),
                                     optimizer=cfg.get("optimizer", "adam"), lr=cfg["lr"], seed=cfg["seed"],
                                     key=j["key"], cohort=c, n=int(msk.sum()), recall=float((pred[msk] == lab).mean()),
                                     tcga_recall=j["test"]["per_class_recall"][lab],
                                     **{f"p_{CLASSES[i]}": float(dist[i]) for i in range(9)}))
        print(fm, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "external_eval.csv", index=False)
    print(df.groupby(["grid", "fm", "algorithm", "optimizer", "lr", "cohort"])[["recall", "tcga_recall"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
