#!/usr/bin/env python
"""
Unit tests and invariants for the FedFM-WSI engine.

  metrics      C-index (lifelines convention), accuracy / balanced accuracy / macro-F1, cluster bootstrap
  cox          Breslow partial likelihood with tied times (brute force, permutation invariance, stratification)
  aggregation  8-statistic slide aggregation reproduces the TCGA slide-feature cache (needs the tile cache)
  cohorts      client partitions and survival cohorts (needs data/*.json, included)
  engine       FL invariants on GPU (needs the TCGA feature cache)
  results      run counts, selection manifest, validation-argmax learning rates (needs results/ and analysis/tables/)

Sections whose inputs are unavailable are skipped. Every check prints PASS / FAIL / SKIP;
the exit status is non-zero if any check fails.
"""
import os, sys, json, glob
from pathlib import Path
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd, torch

T = ROOT / "analysis" / "tables"
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def skip(name, why):
    print(f"SKIP {name}  [{why}]")


def load(n):
    return pd.read_csv(T / f"{n}.csv")


def near(x, y, tol=0.05):
    return abs(x - y) < tol


def test_metrics():
    from fedfm.metrics import harrell_c, classification_metrics, cluster_bootstrap_indices
    from lifelines.utils import concordance_index
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(100):
        n = rng.integers(20, 150)
        risk = rng.integers(0, 10, n).astype(float); time = rng.integers(1, 25, n).astype(float); event = rng.integers(0, 2, n)
        if event.sum() == 0:
            continue
        ok &= abs(harrell_c(risk, time, event) - concordance_index(time, -risk, event)) < 1e-12
    check("harrell_c equals lifelines concordance_index (100 random sets with tied times and tied risks)", ok)
    ok = True
    for _ in range(10):
        y = rng.integers(0, 9, 500); p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 9, 500))
        m = classification_metrics(y, p)
        ok &= abs(m["accuracy"] - accuracy_score(y, p)) < 1e-12 and abs(m["balanced_accuracy"] - balanced_accuracy_score(y, p)) < 1e-12 \
            and abs(m["macro_f1"] - f1_score(y, p, average="macro")) < 1e-12
    check("accuracy / balanced accuracy / macro-F1 equal sklearn", ok)
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    idx = next(cluster_bootstrap_indices(groups, 1, np.random.default_rng(1)))
    check("cluster bootstrap draws whole patients", all(np.isin(np.where(groups == groups[i])[0], idx).all() for i in idx))


def test_cox():
    from fedfm.models import cox_loss, stratified_cox_loss, cox_loss_legacy
    r = torch.tensor([0., 1.]); t = torch.tensor([1., 1.]); e = torch.tensor([1., 1.])
    check("Breslow loss on a tied pair = 0.813262 in both orders",
          near(float(cox_loss(r, t, e)), 0.813262, 1e-5) and near(float(cox_loss(r.flip(0), t.flip(0), e.flip(0))), 0.813262, 1e-5))
    torch.manual_seed(0); n = 300
    r = torch.randn(n); t = torch.randint(1, 40, (n,)).float(); e = (torch.rand(n) < 0.5).float(); st = torch.randint(0, 5, (n,))
    brute = -sum((r[i] - torch.logsumexp(r[t >= t[i]], 0)) * e[i] for i in range(n)) / n
    brute_s = -sum((r[i] - torch.logsumexp(r[(t >= t[i]) & (st == st[i])], 0)) * e[i] for i in range(n)) / n
    perm = torch.randperm(n)
    check("Breslow loss equals brute force and is permutation invariant (300 samples, many ties)",
          near(float(cox_loss(r, t, e)), float(brute), 1e-5) and near(float(cox_loss(r[perm], t[perm], e[perm])), float(brute), 1e-5))
    check("stratified Breslow loss equals brute force", near(float(stratified_cox_loss(r, t, e, st)), float(brute_s), 1e-5))
    per = sum(cox_loss(r[st == s_], t[st == s_], e[st == s_]) * (st == s_).sum() for s_ in range(5)) / n
    check("stratified loss = size-weighted sum of client-local losses", near(float(stratified_cox_loss(r, t, e, st)), float(per), 1e-5))
    a = float(cox_loss_legacy(torch.tensor([0., 1.]), torch.tensor([1., 1.]), torch.tensor([1., 1.])))
    b = float(cox_loss_legacy(torch.tensor([1., 0.]), torch.tensor([1., 1.]), torch.tensor([1., 1.])))
    check("legacy loss is order dependent on tied times (known defect, kept for reproduction only)", not near(a, b, 1e-6))


def test_aggregation():
    from fedfm.data import FEATURE_ROOT
    from build_cptac_features import agg
    tiles_root = Path("/mnt/10t/cached_tiles_npy")
    if not FEATURE_ROOT.exists() or not tiles_root.exists():
        skip("8-statistic aggregation reproduces the TCGA cache", "tile / slide caches not available"); return
    ok = True; nchk = 0
    for fm, sub in [("UNI_v2", "uni_v2"), ("Conch_v15", "conch_v15")]:
        for nm in sorted(os.listdir(FEATURE_ROOT / fm))[:3]:
            tp = tiles_root / sub / nm
            if not tp.exists():
                continue
            S = np.load(FEATURE_ROOT / fm / nm); Tt = np.load(tp)
            ok &= np.abs(agg(Tt) - S).max() < 1e-4; nchk += 1
    check(f"8-statistic aggregation reproduces the TCGA cache ({nchk} slides)", ok and nchk >= 4)


def test_cohorts():
    from fedfm.partitions import client_assignment
    from collections import Counter
    S = json.load(open(ROOT / "data" / "cohort_classification.json"))
    test_ids = [s["client_id"] for s in S if s["split"] == "test"]
    base_pat = Counter(); seen = set()
    for s in S:
        if s["split"] == "train" and s["case_id"] not in seen:
            base_pat[s["client_id"]] += 1; seen.add(s["case_id"])
    for p in ["institution", "labelskew_0.1", "labelskew_1.0", "iid"]:
        co = client_assignment(S, p)
        check(f"partition {p}: test slides keep their Project_TSS id", [c for c, s in zip(co, S) if s["split"] == "test"] == test_ids)
        if p != "institution":
            pat = Counter(); seen = set()
            for c, s in zip(co, S):
                if s["split"] == "train" and s["case_id"] not in seen:
                    pat[c] += 1; seen.add(s["case_id"])
            check(f"partition {p}: training-patient quota per client preserved", sorted(pat.values()) == sorted(base_pat.values()))
    co = client_assignment(S, "institution_all")
    check("institution_all: 58 clients and every test slide assigned to one",
          len(set(co)) == 58 and all(c.startswith("INST:") for c, s in zip(co, S) if s["split"] == "test"))
    for c, n_cl in [("BRCA", 21), ("COAD", 14), ("STAD", 11)]:
        sv = json.load(open(ROOT / "data" / f"cohort_survival_{c}.json"))
        by = {}
        for x in sv:
            by.setdefault(x["client_id"], set()).add(x["case_id"])
        sp = {}
        for x in sv:
            sp.setdefault(x["case_id"], set()).add(x["split"])
        check(f"survival {c}: {n_cl} clients, >= 5 patients each, no patient across splits",
              len(by) == n_cl and min(len(v) for v in by.values()) >= 5 and all(len(v) == 1 for v in sp.values()))
    return S


def test_engine(S):
    from fedfm import data as D
    from fedfm.fl import FLConfig, FLRun
    from fedfm.partitions import client_assignment
    from fedfm.selectors import make_selector
    if not torch.cuda.is_available():
        skip("FL engine invariants", "no CUDA device"); return
    fn = [s["filename"] for s in S]
    try:
        X = torch.tensor(np.asarray(D.load_feature_matrix("Conch_v15", fn)), device="cuda")
    except Exception as ex:  # feature cache not available
        skip("FL engine invariants", f"feature matrix unavailable: {type(ex).__name__}"); return
    y = torch.tensor([s["label"] for s in S], device="cuda")
    co = client_assignment(S, "project_tss")
    run = FLRun(FLConfig(fm="Conch_v15", algorithm="FedAvg", seed=0), X, y, S, co, "cuda"); sel = run.select(1)
    st_ = [run.local_train(c)[0] for c in sel]
    by = {}
    for c in sel:
        by.setdefault(c.label, []).append(c)
    w = [c.n_train / sum(cc.n_train for cc in by[c.label]) / len(by) for c in sel]
    agg_state = run.aggregate(st_, sel); manual = {k: sum(wi * s_[k].float() for wi, s_ in zip(w, st_)) for k in st_[0]}
    check("class-balanced aggregation: weights sum to 1, all 9 classes sampled, aggregate() equals manual average",
          abs(sum(w) - 1) < 1e-9 and len(by) == 9 and all(torch.allclose(agg_state[k].float(), manual[k], atol=1e-6) for k in manual))
    rF = FLRun(FLConfig(fm="Conch_v15", algorithm="FedAvg", seed=3, fixed_rounds=1), X, y, S, co, "cuda"); _, _, sF = rF.run()
    rS = FLRun(FLConfig(fm="Conch_v15", algorithm="SCAFFOLD", seed=3, fixed_rounds=1), X, y, S, co, "cuda"); _, _, sS = rS.run()
    check("SCAFFOLD with zero control variates reproduces FedAvg in round 1", all(torch.allclose(sF[k].float(), sS[k].float(), atol=1e-5) for k in sF))
    stored = sorted(glob.glob(str(ROOT / "results/cls/main/Conch_v15/*.json")))
    if stored:
        j = json.load(open(stored[0]))
        cfg = FLConfig(**{k: (tuple(v) if k == "hidden_dims" else v) for k, v in j["config"].items()})
        rec, _, _ = FLRun(cfg, X, y, S, client_assignment(S, cfg.partition), "cuda").run()
        check("re-running a stored configuration reproduces its test accuracy", abs(rec["test"]["accuracy"] - j["test"]["accuracy"]) < 1e-3,
              f"{rec['test']['accuracy']:.4f} vs {j['test']['accuracy']:.4f}")

    class Spy:
        def __init__(self, inner): self.inner = inner; self.rewards = []
        def select(self, cl, k): return self.inner.select(cl, k)
        def update(self, sel, reward): self.rewards.append(reward); self.inner.update(sel, reward)
    rU = FLRun(FLConfig(fm="Conch_v15", algorithm="FedAvg", sampling="ucb", fixed_rounds=6, seed=0), X, y, S, co, "cuda")
    rU.selector = Spy(make_selector("ucb", rU.clients, X)); recU, _, _ = rU.run()
    h = recU["val_history"]; expect = [max(0.0, h[i] - h[i - 1]) for i in range(1, len(h))]
    check("UCB reward equals max(0, val_t - val_{t-1}) of the pooled validation accuracy", np.allclose(rU.selector.rewards, expect))
    check("coverage and selection counts are logged", len(recU["coverage_history"]) == 6 and sum(recU["selection_counts"].values()) == 120)


def test_results():
    if not (ROOT / "results" / "cls").exists() or not T.exists():
        skip("result counts and selection checks", "results/ or analysis/tables/ missing"); return
    expect = {"main": 140, "sgd": 560, "central": 280, "mu": 105, "lr": 70, "lr_all": 210, "participation": 80, "linear": 140,
              "controls": 140, "partition": 175, "partition_prox": 70, "partition_prox2": 105, "selection": 80, "sgd_mu": 180,
              "legacy_droplast": 70, "partition_fedbn": 40, "partition_fedbn_sgd": 80, "partition_bnref": 60, "controls2": 140, "linear2": 140}
    for g, n in expect.items():
        got = len(glob.glob(str(ROOT / "results/cls" / g / "*/*.json")))
        check(f"grid {g}: {n} runs", got == n, str(got))
    check("survival: 1155 runs", len(glob.glob(str(ROOT / "results/surv/*/*/*.json"))) == 1155)
    man = json.load(open(ROOT / "analysis/selection_manifest.json"))
    check("selection manifest covers the main tables", all(k in man for k in ["main_adam", "main_adam_sel", "main_sgd", "central", "T_mu_adam", "T_lr_adam", "T_participation"]))
    sgd = load("T_main_sgd"); sweep = load("T_sgd_lr_sweep"); ok = True
    for _, r in sgd.iterrows():
        d = sweep[(sweep.fm == r.fm) & (sweep.algorithm == r.algorithm)]; ok &= abs(d.loc[d.val.idxmax()].lr - r.lr) < 1e-12
    check("SGD learning rate per (encoder, algorithm) is the validation argmax", ok)
    cen = load("T_central"); csw = load("T_central_lr_sweep"); ok = True
    for _, r in cen[cen.protocol == "tuned"].iterrows():
        d = csw[(csw.fm == r.fm) & (csw.optimizer == r.optimizer)]; ok &= abs(d.loc[d.val.idxmax()].lr - r.lr) < 1e-12
    check("centralized learning rate per (encoder, optimiser) is the validation argmax", ok)


def test_guards():
    """CPU-only: BN heads reject single-slide clients; every prediction reader follows an archived integer class."""
    from fedfm.fl import FLConfig, FLRun
    tiny = [{"split": sp, "label": 0, "client_id": "one"} for sp in ("train", "val", "test")]
    ok = True
    for alg in ("FedBN", "FedAvgBN"):
        try:
            FLRun(FLConfig(algorithm=alg, hidden_dims=(4,), dropout=0, class_weighting=False), torch.zeros((3, 2)), torch.zeros(3, dtype=torch.long), tiny, ["one"] * 3, "cpu"); ok = False
        except ValueError as e:
            ok &= "at least two training slides" in str(e)
    check("BN heads reject a client with a single training slide", ok)
    import tempfile
    sys.path.insert(0, str(ROOT / "analysis"))
    from predictions import archived_predictions
    tmp = Path(tempfile.mkdtemp()); idx = np.arange(10); pred = np.arange(10) % 9; probs = np.eye(9, dtype=np.float32)[(pred + 1) % 9]
    np.savez(tmp / "r_pred.npz", test_idx=idx, pred=pred, probs=probs)
    check("archived integer class overrides the stored probabilities", (archived_predictions(tmp / "r.json")[1] == pred).all())
    np.savez(tmp / "s_pred.npz", test_idx=idx, probs=probs)
    check("runs without an integer class use the argmax of the probabilities", (archived_predictions(tmp / "s.json")[1] == probs.argmax(1)).all())


if __name__ == "__main__":
    test_metrics(); test_cox(); test_aggregation(); test_guards()
    S = test_cohorts(); test_engine(S); test_results()
    print(f"\n{len(FAILS)} failures")
    sys.exit(1 if FAILS else 0)
