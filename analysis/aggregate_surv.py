#!/usr/bin/env python
"""
Aggregate within-cancer survival results (results/surv/<cancer>/<fm>/<key>.json + _pred.npz).

  * learning rate selected per (cancer, FM, mode) on validation C-index (seed mean);
  * test patient-level C-index: seed mean ± s.d.;
  * two-level bootstrap (B = 2000; seeds of each arm and test patients resampled) of the
    difference FL − central (and FL − stratified central) on the same test patients; the
    patient-only interval and the paired per-seed t-interval are reported alongside;
  * events / censoring per cancer and split.
Outputs analysis/tables/T_surv*.csv
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
RES = REV / "results" / "surv"
OUT = REV / "analysis" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
CANCERS = ["BRCA", "COAD", "STAD"]
B = 2000
rng = np.random.default_rng(0)


import sys
sys.path.insert(0, str(REV / "code"))
from fedfm.metrics import harrell_c   # vectorised, lifelines tie convention (verified in analysis/verify.py)


def load():
    rows = []
    for f in sorted(RES.glob("*/*/*.json")):
        j = json.load(open(f)); c = j["config"]
        rows.append(dict(cancer=c["cancer"], fm=c["fm"], mode=c["mode"], lr=c["lr"], seed=c["seed"],
                         val=j["best_val"], c_pat=j["test_c_patient"], c_slide=j["test_c_slide"],
                         best_round=j["best_round"], n_pat=j["n_test_patients"], n_ev=j["n_test_events"],
                         path=str(f)))
    return pd.DataFrame(rows)


def select_lr(df):
    d = df[df["mode"].isin(["fl", "central", "central_strat"])]
    m = d.groupby(["cancer", "fm", "mode", "lr"])["val"].mean().reset_index()
    best = m.loc[m.groupby(["cancer", "fm", "mode"])["val"].idxmax()][["cancer", "fm", "mode", "lr"]]
    return d.merge(best, on=["cancer", "fm", "mode", "lr"])


def preds(row):
    p = np.load(row.path.replace(".json", "_pred.npz"), allow_pickle=True)
    return p["cases"], p["risk"].astype(float), p["time"].astype(float), p["event"].astype(float)


def paired_boot(runsA, runsB, seed_key=""):
    import zlib
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()) & 0xFFFFFFFF)
    """Two-level bootstrap: resample the seeds of each arm with replacement and the test patients;
    C-index of each drawn seed's risk scores on the drawn patients, averaged over the drawn seeds.
    Returns point, lo, hi (two-level), lo_pat, hi_pat (patients only), CIs of each arm, and the
    paired per-seed t-interval."""
    A = [preds(r) for _, r in runsA.iterrows()]
    Bt = [preds(r) for _, r in runsB.iterrows()]
    cases, _, T, E = A[0]
    for c, _, _, _ in A + Bt:
        assert np.array_equal(c, cases)
    RA = np.stack([a[1] for a in A]); RB = np.stack([b[1] for b in Bt])
    n = len(cases)
    diffs, diffs_pat, cA, cB = [], [], [], []
    for _ in range(B):
        pick = rng.integers(0, n, n)
        ca_all = np.array([harrell_c(r[pick], T[pick], E[pick]) for r in RA])
        cb_all = np.array([harrell_c(r[pick], T[pick], E[pick]) for r in RB])
        sa = rng.integers(0, len(RA), len(RA)); sb = rng.integers(0, len(RB), len(RB))
        diffs.append(ca_all[sa].mean() - cb_all[sb].mean()); diffs_pat.append(ca_all.mean() - cb_all.mean())
        cA.append(ca_all[sa].mean()); cB.append(cb_all[sb].mean())
    cA_pt = np.array([harrell_c(r, T, E) for r in RA]); cB_pt = np.array([harrell_c(r, T, E) for r in RB])
    point = cA_pt.mean() - cB_pt.mean()
    sA = runsA.sort_values("seed").seed.values; sB = runsB.sort_values("seed").seed.values
    common = sorted(set(sA) & set(sB))
    d = np.array([cA_pt[list(runsA.seed).index(s_)] - cB_pt[list(runsB.seed).index(s_)] for s_ in common])
    from scipy import stats
    tq = stats.t.ppf(0.975, len(d) - 1) if len(d) > 1 else np.nan
    t_lo, t_hi = (d.mean() - tq * d.std(ddof=1) / np.sqrt(len(d)), d.mean() + tq * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else (np.nan, np.nan)
    return (point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.percentile(diffs_pat, 2.5), np.percentile(diffs_pat, 97.5),
            np.percentile(cA, 2.5), np.percentile(cA, 97.5), np.percentile(cB, 2.5), np.percentile(cB, 97.5), t_lo, t_hi)


def main():
    df = load()
    print(len(df), "survival runs")
    sel = select_lr(df)
    t = sel.groupby(["cancer", "fm", "mode", "lr"]).agg(n=("seed", "size"), val=("val", "mean"),
                                                        c=("c_pat", "mean"), c_sd=("c_pat", "std"),
                                                        c_slide=("c_slide", "mean"),
                                                        best_round=("best_round", "mean")).reset_index()
    leg = df[df["mode"].isin(["fl_legacy_restore", "central_legacy"])]
    tl = leg.groupby(["cancer", "fm", "mode", "lr"]).agg(n=("seed", "size"), val=("val", "mean"),
                                                         c=("c_pat", "mean"), c_sd=("c_pat", "std"),
                                                         c_slide=("c_slide", "mean"),
                                                         best_round=("best_round", "mean")).reset_index()
    pd.concat([t, tl]).to_csv(OUT / "T_surv_main.csv", index=False)
    df.groupby(["cancer", "fm", "mode", "lr"]).agg(val=("val", "mean"), c=("c_pat", "mean")).reset_index() \
        .to_csv(OUT / "T_surv_lr_sweep.csv", index=False)

    rows = []
    for cancer in CANCERS:
        for fm in FMS:
            fl = sel[(sel.cancer == cancer) & (sel.fm == fm) & (sel["mode"] == "fl")]
            ce = sel[(sel.cancer == cancer) & (sel.fm == fm) & (sel["mode"] == "central")]
            cs = sel[(sel.cancer == cancer) & (sel.fm == fm) & (sel["mode"] == "central_strat")]
            if not (len(fl) and len(ce)):
                continue
            d, lo, hi, lop, hip, alo, ahi, blo, bhi, tlo, thi = paired_boot(fl, ce, seed_key=f"{cancer}|{fm}|pooled")
            row = dict(cancer=cancer, fm=fm, fl=fl.c_pat.mean(), fl_sd=fl.c_pat.std(), fl_lo=alo, fl_hi=ahi,
                       central=ce.c_pat.mean(), central_sd=ce.c_pat.std(), central_lo=blo, central_hi=bhi,
                       diff=d, diff_lo=lo, diff_hi=hi, diff_lo_pat=lop, diff_hi_pat=hip, diff_t_lo=tlo, diff_t_hi=thi,
                       n_pat=int(fl.n_pat.iloc[0]), n_ev=int(fl.n_ev.iloc[0]))
            if len(cs):
                d2, lo2, hi2, lop2, hip2, _, _, slo, shi, tlo2, thi2 = paired_boot(fl, cs, seed_key=f"{cancer}|{fm}|strat")
                row.update(strat=cs.c_pat.mean(), strat_sd=cs.c_pat.std(), strat_lo=slo, strat_hi=shi,
                           diff_strat=d2, diff_strat_lo=lo2, diff_strat_hi=hi2, diff_strat_lo_pat=lop2, diff_strat_hi_pat=hip2)
            rows.append(row)
            print(cancer, fm, f"FL {row['fl']:.3f} central {row['central']:.3f} diff {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "T_surv_paired.csv", index=False)
    if len(R):
        print("FL > central point:", int((R["diff"] > 0).sum()), "/", len(R),
              "| CI excludes 0 (FL better):", int((R.diff_lo > 0).sum()),
              "| CI excludes 0 (central better):", int((R.diff_hi < 0).sum()))

    # events / censoring table
    ev = []
    for cancer in CANCERS:
        s = json.load(open(REV / "data" / f"cohort_survival_{cancer}.json"))
        pats = {}
        for x in s:
            pats[x["case_id"]] = (x["split"], x["os_status"], x["client_id"])
        by_split = defaultdict(lambda: [0, 0])
        for _, (sp, e, _) in pats.items():
            by_split[sp][0] += 1; by_split[sp][1] += e
        per_client = defaultdict(int)
        for _, (sp, e, cid) in pats.items():
            if sp == "train":
                per_client[cid] += 1
        row = dict(cancer=cancer, slides=len(s), patients=len(pats), clients=len(set(x["client_id"] for x in s)),
                   events=sum(v[1] for v in pats.values()),
                   censored_pct=100 * (1 - sum(v[1] for v in pats.values()) / len(pats)),
                   slides_per_patient=len(s) / len(pats),
                   median_train_patients_per_client=float(np.median(list(per_client.values()))))
        for sp in ["train", "val", "test"]:
            row[f"{sp}_patients"], row[f"{sp}_events"] = by_split[sp]
        ev.append(row)
    pd.DataFrame(ev).to_csv(OUT / "T_surv_events.csv", index=False)
    print(pd.DataFrame(ev).to_string())


if __name__ == "__main__":
    main()
