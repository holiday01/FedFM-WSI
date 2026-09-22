#!/usr/bin/env python
"""
Checkpoint-precision diagnostic.

The CPTAC evaluation (scripts/eval_external.py) applies the archived checkpoints of the `main`, `sgd` and `central`
grids.  Checkpoints written before 2026-09-22 hold float16 weights, whereas the TCGA test metrics come from the
predictions archived at training time (float32 model).  This script re-applies every checkpoint that the CPTAC
evaluation uses (raw features, FedBN excluded) to the TCGA test slides on the GPU, casts the weights back to float32
as eval_external.py does, and compares the resulting class with the archived prediction of the same run and with the
training-time confusion matrix.  Output: analysis/tables/T_checkpoint_precision.csv (one row per checkpoint) and
T_checkpoint_precision_summary.csv.  No reported result is derived from this script.
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "code"))
import numpy as np, pandas as pd, torch
from fedfm import data as D
from fedfm.models import MLPClassifier

T = REV / "analysis" / "tables"
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    samples = json.load(open(REV / "data" / "cohort_classification.json"))
    fnames = [s["filename"] for s in samples]; label = np.array([s["label"] for s in samples])
    rows = []
    for fm in FMS:
        X = torch.tensor(np.asarray(D.load_feature_matrix(fm, fnames)), device=dev)
        for grid in ["main", "sgd", "central"]:
            for pt in sorted((REV / "results" / "cls" / grid / fm).glob("*.pt")):
                j = json.load(open(str(pt)[:-3] + ".json")); cfg = j["config"]
                if cfg.get("algorithm") == "FedBN" or cfg.get("feature_transform", "raw") != "raw":
                    continue
                z = np.load(str(pt)[:-3] + "_pred.npz"); idx = z["test_idx"]
                archived = z["pred"].astype(int) if "pred" in z.files else z["probs"].astype(np.float32).argmax(1)
                st = torch.load(pt, map_location=dev)
                dtypes = sorted(set(str(v.dtype) for v in st.values() if v.is_floating_point()))
                m = MLPClassifier(X.shape[1], 9, tuple(cfg.get("hidden_dims", (512, 256))), cfg.get("dropout", 0.3), False).to(dev)
                m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()}); m.eval()
                with torch.no_grad():
                    pred = torch.cat([m(X[torch.as_tensor(idx[s:s + 2048], device=dev)]).argmax(1) for s in range(0, len(idx), 2048)]).cpu().numpy()
                y = label[idx]
                cm = np.zeros((9, 9), int); np.add.at(cm, (y, pred), 1)
                n_vs_json = int(np.abs(cm - np.array(j["test"]["confusion"])).sum() // 2)
                rows.append(dict(grid=grid, fm=fm, key=j["key"], algorithm=cfg.get("algorithm", "central-" + cfg.get("protocol", "")),
                                 optimizer=cfg.get("optimizer", "adam"), lr=cfg["lr"], seed=cfg["seed"], checkpoint_dtype=";".join(dtypes),
                                 acc_json=j["test"]["accuracy"], acc_archived=float((archived == y).mean()), acc_checkpoint=float((pred == y).mean()),
                                 n_diff_vs_archived=int((pred != archived).sum()), n_diff_vs_json=n_vs_json))
        del X; torch.cuda.empty_cache(); print(fm, len(rows), flush=True)
    df = pd.DataFrame(rows); df["shift_vs_archived_pp"] = 100 * (df.acc_checkpoint - df.acc_archived); df["shift_vs_json_pp"] = 100 * (df.acc_checkpoint - df.acc_json)
    df.to_csv(T / "T_checkpoint_precision.csv", index=False)
    summ = dict(n_checkpoints=len(df), dtypes=";".join(sorted(df.checkpoint_dtype.unique())),
                checkpoints_identical_to_archived=int((df.n_diff_vs_archived == 0).sum()), checkpoints_identical_to_json=int((df.n_diff_vs_json == 0).sum()),
                median_diff_slides_vs_archived=float(df.n_diff_vs_archived.median()), max_diff_slides_vs_archived=int(df.n_diff_vs_archived.max()),
                max_abs_shift_vs_archived_pp=float(df.shift_vs_archived_pp.abs().max()), max_abs_shift_vs_json_pp=float(df.shift_vs_json_pp.abs().max()),
                p95_abs_shift_vs_archived_pp=float(df.shift_vs_archived_pp.abs().quantile(0.95)))
    pd.DataFrame([summ]).to_csv(T / "T_checkpoint_precision_summary.csv", index=False)
    print(summ)


if __name__ == "__main__":
    main()
