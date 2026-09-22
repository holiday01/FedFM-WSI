"""Leave-one-site-out diagnostic for the legacy FedFM-WSI survival results.

The benchmark reports within-cancer FL survival C-index (BRCA 0.616, STAD 0.612, COAD 0.538)
under a PATIENT-LEVEL (case_id) split — test patients come from hospitals seen in training.
TCGA site signatures are prognostic (Howard, Nat Commun 2021), so that split can leak the
hospital. This quantifies the leak with the exact foundation models the benchmark uses, and
adds the three baselines the benchmark omits.

For each cancer x FM: C-index under random folds vs entire-sites-held-out (GroupKFold on
TSS), plus:
  site one-hot (no image)   - if this wins, the metric is reading the hospital
  age + stage (no image)    - if this wins, the image adds nothing clinical
  age + stage + image       - does the image add anything on top of clinical?
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tcga  # noqa: E402

ROOT = "/home/holiday01/visium_wsi"
FMS = ["UNI_v2", "Virchow2", "Conch_v15", "Phikon_v2"]


def _reduce(Xtr, Xte, npc, seed, n_raw=0):
    """PCA the image block; pass the first n_raw (clinical) columns through untouched.

    Without this, concatenating 2 clinical dims with a high-dim image block and PCA-ing
    the whole thing washes the clinical signal out — the clin+image arm then just equals
    image-only, which is a methods artefact, not a result.
    """
    Xr_tr, Xr_te = Xtr[:, :n_raw], Xte[:, :n_raw]
    Xi_tr, Xi_te = Xtr[:, n_raw:], Xte[:, n_raw:]
    s0 = StandardScaler().fit(Xi_tr)
    Xi_tr, Xi_te = s0.transform(Xi_tr), s0.transform(Xi_te)
    k = min(npc, Xi_tr.shape[1], len(Xtr) // 4)
    if Xi_tr.shape[1] > k and k > 0:
        p = PCA(k, random_state=seed).fit(Xi_tr)
        Xi_tr, Xi_te = p.transform(Xi_tr), p.transform(Xi_te)
    Ztr = np.c_[Xr_tr, Xi_tr] if n_raw else Xi_tr
    Zte = np.c_[Xr_te, Xi_te] if n_raw else Xi_te
    s1 = StandardScaler().fit(Ztr)
    return s1.transform(Ztr), s1.transform(Zte)


def cv(X, T, E, groups=None, strat=None, npc=32, seeds=(0, 1, 2), n_raw=0):
    sc = []
    for s in seeds:
        sp = (GroupKFold(5).split(X, T, groups) if groups is not None
              else StratifiedKFold(5, shuffle=True, random_state=s).split(X, strat))
        for tr, te in sp:
            if E[tr].sum() < 5 or E[te].sum() < 3:
                continue
            Xtr, Xte = _reduce(X[tr], X[te], npc, s, n_raw=n_raw)
            c = [f"p{i}" for i in range(Xtr.shape[1])]
            df = pd.DataFrame(Xtr, columns=c)
            df["T"], df["E"] = T[tr], E[tr]
            try:
                m = CoxPHFitter(penalizer=0.5).fit(df, "T", "E")
            except Exception:
                continue
            r = -m.predict_partial_hazard(pd.DataFrame(Xte, columns=c)).to_numpy()
            sc.append(concordance_index(T[te], r, E[te]))
        if groups is not None:
            break
    return (float(np.mean(sc)), float(np.std(sc))) if sc else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/F_fedfm_loso_diagnostic.csv")
    args = ap.parse_args()

    clin = tcga.load_clinical()
    rows = []
    for cancer in ["TCGA-BRCA", "TCGA-COAD", "TCGA-STAD"]:
        cc = clin[clin.Project == cancer].dropna(subset=["age_at_index"])
        st = cc.ajcc_pathologic_stage.astype(str)
        late = st.str.contains(r"Stage\s+(?:III|IV)", regex=True, na=False).to_numpy().astype(float)

        # baselines shared by all FMs on the largest common patient set
        base_pats = set(cc.index)
        for fm in FMS:
            p, _ = tcga.baseline_patient_matrix(fm)
            base_pats &= set(p)
        common0 = sorted(base_pats)
        c0 = cc.loc[common0]
        T0, E0 = c0.OS_time.to_numpy(float), c0.OS_status.to_numpy(float)
        site0 = np.array([p.split("-")[1] for p in common0])
        AGE = c0.age_at_index.to_numpy(float)
        LATE = late[cc.index.get_indexer(common0)]
        CLIN = np.c_[AGE, LATE]
        SITE1H = pd.get_dummies(pd.Series(site0)).to_numpy(float)

        print(f"\n{'='*70}\n{cancer}: n={len(common0)}, {len(set(site0))} sites\n{'='*70}", flush=True)
        m, _ = cv(SITE1H, T0, E0, strat=E0)
        rows.append({"cancer": cancer, "arm": "NO-IMAGE: site one-hot", "C_random": m, "C_loso": np.nan})
        print(f"  NO-IMAGE site one-hot   random={m:.3f}", flush=True)
        m, _ = cv(CLIN, T0, E0, strat=E0)
        ml, _ = cv(CLIN, T0, E0, groups=site0)
        rows.append({"cancer": cancer, "arm": "NO-IMAGE: age+stage", "C_random": m, "C_loso": ml})
        print(f"  NO-IMAGE age+stage      random={m:.3f}  loso={ml:.3f}", flush=True)

        for fm in FMS:
            pats, X = tcga.baseline_patient_matrix(fm)
            common = sorted(set(cc.index) & set(pats))
            c = cc.loc[common]
            T, E = c.OS_time.to_numpy(float), c.OS_status.to_numpy(float)
            site = np.array([p.split("-")[1] for p in common])
            Xc = X[pd.Index(pats).get_indexer(common)]
            LATEc = late[cc.index.get_indexer(common)]
            CLINc = np.c_[c.age_at_index.to_numpy(float), LATEc]

            m_r, _ = cv(Xc, T, E, strat=E)
            m_l, _ = cv(Xc, T, E, groups=site)
            # clinical dims pass through PCA untouched (n_raw=2), so they are not washed out
            m_ci, _ = cv(np.c_[CLINc, Xc], T, E, groups=site, n_raw=CLINc.shape[1])
            rows.append({"cancer": cancer, "arm": f"IMAGE: {fm}", "C_random": m_r, "C_loso": m_l,
                         "C_loso_clin_plus_image": m_ci})
            print(f"  IMAGE {fm:12s}   random={m_r:.3f}  loso={m_l:.3f}  "
                  f"(drop {m_l-m_r:+.3f})   clin+image_loso={m_ci:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
