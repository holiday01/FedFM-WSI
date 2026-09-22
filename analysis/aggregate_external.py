#!/usr/bin/env python
"""
Paired difference FL (FedAvg, SGD, validation-selected learning rate) minus centralized (SGD, validation-tuned
learning rate) on the CPTAC cohorts, per encoder and cohort, from the per-slide predictions written by
scripts/eval_external.py (results/external/external_preds.npz):
  * two-level bootstrap (B = 2000): training seeds of each arm resampled with replacement and CPTAC patients resampled
    with replacement (one slide per patient in both collections), recall difference recomputed per draw;
  * paired-seed t-interval (seed i against seed i, n = 5).
Reads the selected learning rates from analysis/tables/T_main_sgd.csv and T_central.csv (aggregate_cls.py).
Writes analysis/tables/T_external_diff.csv; analysis/make_tables.py turns it into the LaTeX table.
"""
import zlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

REV = Path(__file__).resolve().parents[1]
T = REV / "analysis" / "tables"
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
B = 2000


def main():
    p = REV / "results" / "external" / "external_eval.csv"; q = REV / "results" / "external" / "external_preds.npz"
    E = pd.read_csv(p); Z = np.load(q, allow_pickle=True)
    sgd = pd.read_csv(T / "T_main_sgd.csv"); cen = pd.read_csv(T / "T_central.csv")
    coh_all = Z["cohort"]; y_all = Z["y"]
    lf = sgd[sgd.algorithm == "FedAvg"].set_index("fm").lr
    lc = cen[(cen.protocol == "tuned") & (cen.optimizer == "sgd")].set_index("fm").lr
    rows = []
    for fm in FMS:
        for coh in ["luad", "pda"]:
            a = E[(E.grid == "sgd") & (E.algorithm == "FedAvg") & (E.fm == fm) & (E.cohort == coh) & (abs(E.lr - lf.get(fm, -1)) < 1e-9)].sort_values("seed")
            c = E[(E.grid == "central") & (E.optimizer == "sgd") & (E.fm == fm) & (E.cohort == coh) & (abs(E.lr - lc.get(fm, -1)) < 1e-9)].sort_values("seed")
            if not len(a) or not len(c):
                continue
            msk = coh_all == coh; lab = int(y_all[msk][0])
            MA = np.array([(Z[k][msk] == lab) for k in a.key]).astype(float); MC = np.array([(Z[k][msk] == lab) for k in c.key]).astype(float)
            assert np.allclose(MA.mean(1), a.recall.values) and np.allclose(MC.mean(1), c.recall.values), "external_eval.csv and external_preds.npz disagree"
            rng = np.random.default_rng(zlib.crc32(f"external|{fm}|{coh}".encode()) & 0xFFFFFFFF); n = MA.shape[1]; est = np.empty(B)
            for b in range(B):
                ip = rng.integers(0, n, n); ia = rng.integers(0, len(MA), len(MA)); ic = rng.integers(0, len(MC), len(MC))
                est[b] = MA[ia][:, ip].mean() - MC[ic][:, ip].mean()
            d = a.recall.values - c.recall.values; m = d.mean(); lo, hi = np.percentile(est, [2.5, 97.5])
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(fm=fm, cohort=coh, fl=a.recall.mean(), fl_sd=a.recall.std(), central=c.recall.mean(), central_sd=c.recall.std(),
                             diff=m, lo=lo, hi=hi, t_lo=m - h, t_hi=m + h, n_patients=int(n), n_seed_pairs=len(d)))
    out = pd.DataFrame(rows); out.to_csv(T / "T_external_diff.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
