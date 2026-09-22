#!/usr/bin/env python
"""
Aggregate classification results (results/cls/<grid>/<fm>/<key>.json + _pred.npz) into
summary tables.  Every selection step (learning rate, mu,
participation) uses the pooled VALIDATION accuracy averaged over seeds; test accuracy is
then reported for the selected configuration.  The selected configurations are
written to analysis/selection_manifest.json.

Uncertainty (v2):
  * seed s.d. (n = 5 or 10 seeds) for every cell;
  * head-to-head differences: TWO-LEVEL bootstrap (B = 2000) that resamples the training
    seeds of each arm with replacement AND the test patients (cluster bootstrap), so the
    interval covers training randomness and test-set sampling;
  * the same patient draws are used for every encoder, so the mean difference over
    encoders has a proper interval;
  * in addition, the paired per-seed differences (seed i vs seed i) are summarised as
    mean, s.d. and a t-interval (n = 5);
  * the conditional patient-only bootstrap (v1) is kept as *_pat columns.

Prediction source:
  * EVERY test metric (accuracy, balanced accuracy, macro-F1, per-class recall, confusion
    matrix, per-client accuracy, per-training-client macro accuracy) is recomputed here from
    the archived per-slide predictions of each run (<key>_pred.npz), the same arrays the
    bootstrap resamples, so tables, figures and intervals share one source.  Runs archived
    before 2026-09-22 store float16 class probabilities only; their class is the argmax of
    those probabilities.  Later runs also store the integer class assigned at evaluation
    (`pred`), which is used when present.
  * The training-time metrics recorded in the run JSON (float32 logits) are kept as
    *_json columns and compared in T_prediction_source_audit.csv /
    T_prediction_source_summary.csv; `best_val` (the validation monitor) comes from the JSON
    because validation predictions are not archived.
"""
import json, sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

REV = Path(__file__).resolve().parents[1]
RES = REV / "results" / "cls"
OUT = REV / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REV / "code"))
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
PATH_FMS = FMS[:-1]
CLASSES = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]
B = 2000
rng = np.random.default_rng(0)

samples = json.load(open(REV / "data" / "cohort_classification.json"))
case_of = np.array([s["case_id"] for s in samples])
label_of = np.array([s["label"] for s in samples])
test_idx_all = np.where(np.array([s["split"] for s in samples]) == "test")[0]
G_uniq, G_inv = np.unique(case_of[test_idx_all], return_inverse=True)
G_cnt = np.bincount(G_inv, minlength=len(G_uniq)).astype(float)
train_size = defaultdict(int)
for s in samples:
    if s["split"] == "train":
        train_size[s["client_id"]] += 1
MANIFEST = {}
from fedfm.metrics import classification_metrics
from fedfm.partitions import client_assignment
client_id_of = np.array([s["client_id"] for s in samples])
_CLIENT_OF = {}


def client_of_partition(partition):
    if partition not in _CLIENT_OF:
        _CLIENT_OF[partition] = np.array(client_assignment(samples, partition))
    return _CLIENT_OF[partition]


def archived_predictions(json_path):
    """(test_idx, predicted class) of a run from its archived prediction file."""
    p = np.load(json_path.replace(".json", "_pred.npz"))
    idx = p["test_idx"]
    pred = p["pred"].astype(int) if "pred" in p.files else p["probs"].astype(np.float32).argmax(1)
    return idx, pred


def archived_metrics(json_path, partition, is_fl):
    """Test metrics recomputed from the archived predictions with the same function the engine
    uses at training time; per-training-client macro accuracy uses the run's own partition."""
    idx, pred = archived_predictions(json_path)
    y = label_of[idx]
    met = classification_metrics(y, pred, client_id_of[idx])
    if is_fl:
        co = client_of_partition(partition)[idx]
        by = defaultdict(lambda: [0, 0])
        for c, t, pr in zip(co, y, pred):
            by[c][0] += int(t == pr); by[c][1] += 1
        met["per_train_client_macro_accuracy"] = float(np.mean([v[0] / v[1] for v in by.values()]))
    return met


def load_all():
    rows = []
    for f in sorted(RES.glob("*/*/*.json")):
        j = json.load(open(f))
        c = j["config"]
        tj = j["test"]                                                   # training-time (float32) evaluation
        kind = "central" if "protocol" in c else "fl"
        t = archived_metrics(str(f), c.get("partition", "project_tss"), kind == "fl")   # archived predictions
        n_changed = int(np.abs(np.array(t["confusion"]) - np.array(tj["confusion"])).sum() // 2)
        r = dict(grid=j.get("grid", f.parts[-3]), fm=f.parts[-2], key=j["key"], path=str(f),
                 kind=kind, acc_json=tj["accuracy"], bacc_json=tj["balanced_accuracy"], f1_json=tj["macro_f1"],
                 n_changed_predictions=n_changed,
                 algorithm=c.get("algorithm", "central-" + c.get("protocol", "")),
                 optimizer=c.get("optimizer", "adam"), lr=c["lr"], mu=c.get("mu"),
                 k=c.get("clients_per_round", -1), sampling=c.get("sampling"),
                 aggregation=c.get("aggregation"), partition=c.get("partition", "project_tss"),
                 ft=c.get("feature_transform", "raw"), arch=c.get("arch", "mlp"),
                 legacy_drop_last=c.get("legacy_drop_last", False), protocol=c.get("protocol"),
                 seed=c["seed"], best_val=j["best_val"], acc=t["accuracy"], bacc=t["balanced_accuracy"],
                 f1=t["macro_f1"], macro=t.get("per_client_macro_accuracy"), train_macro=t.get("per_train_client_macro_accuracy"),
                 rounds=j.get("rounds_run", j.get("epochs_run")),
                 best_round=j.get("best_round", j.get("best_epoch")), n_params=j["n_params"],
                 total_steps=j.get("total_local_steps", j.get("total_steps")),
                 seconds=j["seconds"], recall=t["per_class_recall"], confusion=t["confusion"],
                 recall_LUAD=t["per_class_recall"][4], recall_PAAD=t["per_class_recall"][8],
                 per_client=t.get("per_client"), n_clients=j.get("n_clients"),
                 partition_summary=j.get("partition_summary"),
                 coverage=j.get("coverage_history"), sel_counts=j.get("selection_counts"), val_hist=j.get("val_history"))
        r["k"] = -1 if r["k"] is None else r["k"]
        rows.append(r)
    return pd.DataFrame(rows)


def correctness(row):
    p = np.load(row["path"].replace(".json", "_pred.npz"))
    idx, probs = p["test_idx"], p["probs"].astype(np.float32)
    v = np.full(len(samples), np.nan)
    v[idx] = (probs.argmax(1) == label_of[idx]).astype(float)
    return v[test_idx_all]


def matrix(df):
    df = df.sort_values("seed")
    return np.stack([correctness(r) for _, r in df.iterrows()]), list(df.seed.values)


def per_patient(M):
    return np.stack([np.bincount(G_inv, weights=row, minlength=len(G_uniq)) for row in M])


def two_level(pairs, B=B, seed_key=""):
    """pairs: list of (MA, MB) seed x slide matrices on the same test slides.  Same patient draws
    for every pair; seeds drawn independently per arm and pair.  Returns per-pair
    (diff, lo, hi, lo_pat, hi_pat) and (mean over pairs, lo, hi).  The random stream is seeded
    from `seed_key` so that every comparison is reproducible regardless of call order."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()) & 0xFFFFFFFF)
    PA = [per_patient(a) for a, _ in pairs]; PB = [per_patient(b) for _, b in pairs]
    est = np.zeros((B, len(pairs))); est_pat = np.zeros((B, len(pairs)))
    for b in range(B):
        pick = rng.integers(0, len(G_uniq), len(G_uniq)); c = G_cnt[pick].sum()
        for j, (a, bm) in enumerate(pairs):
            sa = rng.integers(0, a.shape[0], a.shape[0]); sb = rng.integers(0, bm.shape[0], bm.shape[0])
            est[b, j] = PA[j][sa][:, pick].mean(0).sum() / c - PB[j][sb][:, pick].mean(0).sum() / c
            est_pat[b, j] = PA[j][:, pick].mean(0).sum() / c - PB[j][:, pick].mean(0).sum() / c
    point = [float(a.mean() - bm.mean()) for a, bm in pairs]
    per = [(point[j], *np.percentile(est[:, j], [2.5, 97.5]), *np.percentile(est_pat[:, j], [2.5, 97.5])) for j in range(len(pairs))]
    m = est.mean(1)
    return per, (float(np.mean(point)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def seed_paired(MA, sA, MB, sB):
    common = sorted(set(sA) & set(sB))
    if len(common) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0
    d = np.array([MA[sA.index(s)].mean() - MB[sB.index(s)].mean() for s in common])
    n = len(d); t = stats.t.ppf(0.975, n - 1)
    return float(d.mean()), float(d.std(ddof=1)), float(d.mean() - t * d.std(ddof=1) / np.sqrt(n)), float(d.mean() + t * d.std(ddof=1) / np.sqrt(n)), n


def diff_table(groupsA, groupsB, labels, seed_key=""):
    pairs, seeds = [], []
    for a, bm in zip(groupsA, groupsB):
        MA, sA = matrix(a); MB, sB = matrix(bm); pairs.append((MA, MB)); seeds.append((MA, sA, MB, sB))
    per, mean_ci = two_level(pairs, seed_key=seed_key + "|" + "|".join(map(str, labels)))
    rows = []
    for lab, (d, lo, hi, lop, hip), sp in zip(labels, per, seeds):
        sm, ssd, tlo, thi, n = seed_paired(*sp)
        rows.append(dict(label=lab, diff=d, lo=lo, hi=hi, lo_pat=lop, hi_pat=hip, seed_diff_mean=sm, seed_diff_sd=ssd, seed_t_lo=tlo, seed_t_hi=thi, n_seed_pairs=n))
    return rows, mean_ci


def summ(df, by):
    g = df.groupby(by)
    return g.agg(n=("seed", "size"), val=("best_val", "mean"),
                 acc=("acc", "mean"), acc_sd=("acc", "std"),
                 bacc=("bacc", "mean"), bacc_sd=("bacc", "std"),
                 f1=("f1", "mean"), f1_sd=("f1", "std"),
                 macro=("macro", "mean"), macro_sd=("macro", "std"), train_macro=("train_macro", "mean"),
                 rounds=("rounds", "mean"), best_round=("best_round", "mean"), total_steps=("total_steps", "mean"),
                 n_params=("n_params", "first"), keys=("key", lambda x: ";".join(sorted(x)))).reset_index()


def select_by_val(df, by, hp):
    m = df.groupby(by + [hp])["best_val"].mean().reset_index()
    best = m.loc[m.groupby(by)["best_val"].idxmax()][by + [hp]]
    return df.merge(best, on=by + [hp])


def mean_recall(df):
    R = np.array([[np.nan if x is None else x for x in r] for r in df["recall"]])
    return np.nanmean(R, axis=0), np.nanstd(R, axis=0)


def train_client_macro_from_pred(row, client_of):
    p = np.load(row["path"].replace(".json", "_pred.npz"))
    idx, probs = p["test_idx"], p["probs"].astype(np.float32)
    pred = probs.argmax(1)
    by = defaultdict(lambda: [0, 0])
    for i, pr in zip(idx, pred):
        by[client_of[i]][0] += int(pr == label_of[i]); by[client_of[i]][1] += 1
    return float(np.mean([v[0] / v[1] for v in by.values()]))


def main():
    df = load_all()
    print(f"{len(df)} runs loaded; grids: {df.grid.value_counts().to_dict()}")
    S = {}
    E = lambda: df.iloc[0:0]

    # 0. per-run table (used by the figures) and the prediction-source audit
    df["acc_shift_pp"] = 100 * (df.acc - df.acc_json)
    df[["grid", "fm", "key", "kind", "algorithm", "optimizer", "lr", "mu", "k", "sampling", "partition", "ft", "arch",
        "legacy_drop_last", "protocol", "seed", "best_val", "acc", "bacc", "f1", "macro", "train_macro", "rounds", "best_round",
        "acc_json", "bacc_json", "f1_json", "n_changed_predictions", "acc_shift_pp", "recall_LUAD", "recall_PAAD"]].to_csv(OUT / "T_runs.csv", index=False)
    print(f"prediction-source audit: {int((df.n_changed_predictions > 0).sum())} of {len(df)} runs differ from the training-time "
          f"confusion matrix; max |shift| {df.acc_shift_pp.abs().max():.4f} pp")

    # 1. main protocol (Adam 3e-4 fixed, as in the legacy code)
    main = df[df.grid == "main"]
    t = summ(main, ["fm", "algorithm"]); t.to_csv(OUT / "T_main_adam.csv", index=False); S["main_adam"] = t.to_dict("records")
    MANIFEST["main_adam"] = {f"{r.fm}|{r.algorithm}": r.keys for r in t.itertuples()}

    # 1b. Adam with the learning rate selected on validation per (fm, algorithm)
    adam_all = df[(df.kind == "fl") & (df.optimizer == "adam") & (df.grid.isin(["main", "lr", "lr_all"]))]
    adam_sel = select_by_val(adam_all, ["fm", "algorithm"], "lr") if len(adam_all) else E()
    if len(adam_sel):
        t = summ(adam_sel, ["fm", "algorithm", "lr"]); t.to_csv(OUT / "T_main_adam_sel.csv", index=False); S["main_adam_sel"] = t.to_dict("records")
        MANIFEST["main_adam_sel"] = {f"{r.fm}|{r.algorithm}": dict(lr=r.lr, keys=r.keys) for r in t.itertuples()}
        summ(adam_all, ["fm", "algorithm", "lr"]).to_csv(OUT / "T_adam_lr_sweep_all.csv", index=False)

    # 2. SGD control
    sgd = df[df.grid == "sgd"]
    sel = select_by_val(sgd, ["fm", "algorithm"], "lr") if len(sgd) else E()
    if len(sel):
        t = summ(sel, ["fm", "algorithm", "lr"]); t.to_csv(OUT / "T_main_sgd.csv", index=False); S["main_sgd"] = t.to_dict("records")
        MANIFEST["main_sgd"] = {f"{r.fm}|{r.algorithm}": dict(lr=r.lr, keys=r.keys) for r in t.itertuples()}
        summ(sgd, ["fm", "algorithm", "lr"]).to_csv(OUT / "T_sgd_lr_sweep.csv", index=False)

    # 3. centralised (Adam / SGD tuned, legacy) and matched-optimiser FL gaps
    cen = df[df.grid == "central"]
    tuned = E()
    if len(cen):
        tuned = select_by_val(cen[cen.protocol == "tuned"], ["fm", "optimizer"], "lr")
        t = summ(tuned, ["fm", "optimizer", "lr"]); t["protocol"] = "tuned"
        leg = summ(cen[cen.protocol == "legacy"], ["fm"]); leg["protocol"] = "legacy"; leg["optimizer"] = "adam"
        cen_t = pd.concat([t, leg]); cen_t.to_csv(OUT / "T_central.csv", index=False); S["central"] = cen_t.to_dict("records")
        summ(cen[cen.protocol == "tuned"], ["fm", "optimizer", "lr"]).to_csv(OUT / "T_central_lr_sweep.csv", index=False)
        MANIFEST["central"] = {f"{r.fm}|{r.optimizer}": dict(lr=r.lr, keys=r.keys) for r in t.itertuples()}
        gaps = []
        for name, fl_df, c_opt in [("sgd", sel, "sgd"), ("adam", main, "adam"), ("adam_sel", adam_sel, "adam")]:
            A, Bm, labs = [], [], []
            for fm in FMS:
                c = tuned[(tuned.fm == fm) & (tuned.optimizer == c_opt)]
                f = fl_df[(fl_df.fm == fm) & (fl_df.algorithm == "FedAvg")] if len(fl_df) else E()
                if len(c) and len(f):
                    A.append(c); Bm.append(f); labs.append(fm)
            if not labs:
                continue
            rows, _ = diff_table(A, Bm, labs, seed_key=f"gap:{name}")
            ip = [i for i, l in enumerate(labs) if l != "ResNet50"]
            _, mean_ci = diff_table([A[i] for i in ip], [Bm[i] for i in ip], [labs[i] for i in ip], seed_key=f"gap:{name}")
            for r in rows:
                fm = r.pop("label")
                r.update(fm=fm, optimizer=name, fl_acc=float(fl_df[(fl_df.fm == fm) & (fl_df.algorithm == "FedAvg")].acc.mean()),
                         central_acc=float(tuned[(tuned.fm == fm) & (tuned.optimizer == c_opt)].acc.mean()))
                gaps.append(r)
            gaps.append(dict(fm="MEAN_PATHOLOGY", optimizer=name, diff=mean_ci[0], lo=mean_ci[1], hi=mean_ci[2]))
        pd.DataFrame(gaps).to_csv(OUT / "T_fl_gap.csv", index=False); S["fl_gap"] = gaps

    # 3b. prediction-source audit for the cells of the main tables (mean over seeds of archived - training-time accuracy)
    aud = []
    for name, sub, by in [("T_main_adam", main, ["fm", "algorithm"]), ("T_main_adam_sel", adam_sel, ["fm", "algorithm"]),
                          ("T_main_sgd", sel, ["fm", "algorithm"]), ("T_central", tuned, ["fm", "optimizer"])]:
        if not len(sub): continue
        for keys, g in sub.groupby(by):
            aud.append(dict(table=name, fm=keys[0], setting=keys[1], n_runs=len(g), affected_runs=int((g.n_changed_predictions > 0).sum()),
                            mean_shift_pp=float(g.acc_shift_pp.mean()), max_abs_run_shift_pp=float(g.acc_shift_pp.abs().max())))
    aud = pd.DataFrame(aud); aud.to_csv(OUT / "T_prediction_source_audit.csv", index=False)
    pa = aud[aud.fm != "ResNet50"]
    cell_key = ["grid", "fm", "algorithm", "optimizer", "lr", "mu", "k", "sampling", "partition", "ft", "arch", "legacy_drop_last", "protocol"]
    anyc = df.fillna({"mu": -1, "k": -1, "sampling": "", "partition": "", "protocol": ""}).groupby(cell_key, dropna=False).acc_shift_pp.mean()
    worst = anyc.abs().idxmax()
    pd.DataFrame([dict(n_runs=len(df), runs_confusion_differs=int((df.n_changed_predictions > 0).sum()),
                       max_abs_cell_shift_any_table_pp=float(anyc.abs().max()), worst_cell=" ".join(str(x) for x in worst[:5]),
                       max_abs_cell_shift_any_table_pathology_pp=float(anyc[[k for k in anyc.index if k[1] != "ResNet50"]].abs().max()),
                       runs_accuracy_differs=int((df.acc_shift_pp.abs() > 1e-9).sum()),
                       max_abs_run_shift_pp=float(df.acc_shift_pp.abs().max()), max_changed_predictions=int(df.n_changed_predictions.max()),
                       max_abs_cell_shift_pathology_pp=float(pa.mean_shift_pp.abs().max()), max_abs_cell_shift_all_pp=float(aud.mean_shift_pp.abs().max()),
                       cells=len(aud), cells_affected=int((aud.affected_runs > 0).sum()))]).to_csv(OUT / "T_prediction_source_summary.csv", index=False)

    # 4. algorithm vs FedAvg
    diffs = []
    for name, sub in [("adam", main), ("adam_sel", adam_sel), ("sgd", sel)]:
        if not len(sub):
            continue
        for algo in ["FedProx", "SCAFFOLD", "FedBN"]:
            A, Bm, labs = [], [], []
            for fm in FMS:
                o = sub[(sub.fm == fm) & (sub.algorithm == algo)]; base = sub[(sub.fm == fm) & (sub.algorithm == "FedAvg")]
                if len(o) and len(base):
                    A.append(o); Bm.append(base); labs.append(fm)
            if not labs:
                continue
            rows, _ = diff_table(A, Bm, labs, seed_key=f"algo:{name}:{algo}")
            ip = [i for i, l in enumerate(labs) if l != "ResNet50"]
            _, mean_ci = diff_table([A[i] for i in ip], [Bm[i] for i in ip], [labs[i] for i in ip], seed_key=f"algo:{name}:{algo}")
            for r in rows:
                r.update(fm=r.pop("label"), optimizer=name, algorithm=algo); diffs.append(r)
            diffs.append(dict(fm="MEAN_PATHOLOGY", optimizer=name, algorithm=algo, diff=mean_ci[0], lo=mean_ci[1], hi=mean_ci[2]))
    pd.DataFrame(diffs).to_csv(OUT / "T_algo_vs_fedavg.csv", index=False); S["algo_vs_fedavg"] = diffs

    # 5. encoder ranking (FedAvg); restricted ranking excludes encoders with KNOWN TCGA inclusion
    KNOWN_TCGA = ["Phikon_v2", "CTransPath", "Midnight12k"]
    rank = []
    for name, sub in [("adam", main), ("adam_sel", adam_sel), ("sgd", sel)]:
        fa = sub[sub.algorithm == "FedAvg"] if len(sub) else E()
        if not len(fa):
            continue
        t = summ(fa, ["fm"]).sort_values("acc", ascending=False).reset_index(drop=True)
        t["optimizer"] = name; t["rank_all"] = np.arange(1, len(t) + 1)
        r = 1; t["rank_no_known_tcga"] = None
        for i, row in t.iterrows():
            if row.fm not in KNOWN_TCGA:
                t.at[i, "rank_no_known_tcga"] = r; r += 1
        best = t.iloc[0].fm
        rows, _ = diff_table([fa[fa.fm == fm] for fm in t.fm], [fa[fa.fm == best] for fm in t.fm], list(t.fm), seed_key=f"rank:{name}")
        for i, r_ in enumerate(rows):
            for k_ in ["diff", "lo", "hi", "lo_pat", "hi_pat", "seed_diff_mean", "seed_diff_sd", "seed_t_lo", "seed_t_hi"]:
                t.at[i, f"vs_best_{k_}"] = r_[k_]
        rank.append(t)
    if rank:
        pd.concat(rank).to_csv(OUT / "T_fm_ranking.csv", index=False)

    # 6. sweeps (validation-selected)
    for grid, hp, fname in [("mu", "mu", "T_mu_adam.csv"), ("lr", "lr", "T_lr_adam.csv"),
                            ("participation", "k", "T_participation.csv"), ("sgd_mu", "mu", "T_mu_sgd.csv")]:
        g = df[df.grid == grid]
        if grid == "mu":
            g = pd.concat([g, main[main.algorithm == "FedProx"]])
        if grid == "lr":
            g = pd.concat([g, main[main.algorithm == "FedAvg"]])
        if grid == "participation":
            g = pd.concat([g, main[(main.algorithm == "FedAvg") & (main.fm.isin(g.fm.unique()))]])
        if grid == "sgd_mu" and len(sgd):
            g = pd.concat([g, sgd[(sgd.algorithm == "FedProx") & (sgd.fm.isin(g.fm.unique()))]])
        if not len(g):
            continue
        by = ["fm", hp] if grid != "sgd_mu" else ["fm", "lr", hp]
        t = summ(g, by)
        sel_hp = t.loc[t.groupby("fm")["val"].idxmax()][["fm"] + by[1:]]; sel_hp["selected"] = True
        t = t.merge(sel_hp, on=["fm"] + by[1:], how="left").fillna({"selected": False})
        t.to_csv(OUT / fname, index=False); S[fname[:-4]] = t.to_dict("records")
        MANIFEST[fname[:-4]] = {r.fm: {h: float(getattr(r, h)) for h in by[1:]} for r in t[t.selected].itertuples()}

    # 7. partitions
    part = df[df.grid.isin(["partition", "partition_prox", "partition_prox2"])]
    if len(part):
        t = summ(part, ["fm", "algorithm", "partition"])
        ps = part.groupby("partition")["partition_summary"].first()
        t["n_clients"] = t.partition.map(lambda p: ps[p]["n_clients"])
        t["classes_per_client"] = t.partition.map(lambda p: ps[p]["mean_classes_per_client"])
        t["majority_fraction"] = t.partition.map(lambda p: ps[p]["mean_majority_fraction"])
        t.to_csv(OUT / "T_partition.csv", index=False); S["partition"] = t.to_dict("records")

    # 8. controls, standardised follow-ups, linear probes
    ctrl = df[df.grid == "controls"]
    if len(ctrl) and len(tuned):
        allc = pd.concat([ctrl, main[main.algorithm == "FedAvg"].assign(ft="raw"), tuned[tuned.optimizer == "adam"].assign(ft="raw")])
        t = summ(allc, ["kind", "fm", "ft"]); t.to_csv(OUT / "T_controls.csv", index=False); S["controls"] = t.to_dict("records")
        from scipy.stats import kendalltau
        taus = []
        for kind in ["fl", "central"]:
            r0 = t[(t.kind == kind) & (t.ft == "raw")].set_index("fm")["acc"]
            for ft in ["zscore", "zscore_rp"]:
                r1 = t[(t.kind == kind) & (t.ft == ft)].set_index("fm")["acc"]
                common = [f for f in FMS if f in r0.index and f in r1.index]
                if len(common) > 2:
                    taus.append(dict(kind=kind, ft=ft, tau=kendalltau(r0[common], r1[common]).statistic, n=len(common)))
        pd.DataFrame(taus).to_csv(OUT / "T_controls_kendall.csv", index=False)
    c2 = df[df.grid == "controls2"]
    if len(c2):
        c2s = pd.concat([select_by_val(c2[c2.optimizer == "sgd"], ["fm", "algorithm"], "lr"), c2[c2.optimizer == "adam"]])
        summ(c2s, ["fm", "algorithm", "optimizer", "lr"]).to_csv(OUT / "T_controls2.csv", index=False)
    lin = df[df.grid == "linear"]
    if len(lin) and len(tuned):
        lin_c = select_by_val(lin[lin.kind == "central"], ["fm"], "lr")
        t = pd.concat([summ(lin[lin.kind == "fl"], ["fm", "arch", "kind"]), summ(lin_c, ["fm", "arch", "kind"]),
                       summ(main[main.algorithm == "FedAvg"], ["fm", "arch", "kind"]), summ(tuned[tuned.optimizer == "adam"], ["fm", "arch", "kind"])])
        t.to_csv(OUT / "T_linear_vs_mlp.csv", index=False); S["linear"] = t.to_dict("records")
    l2 = df[df.grid == "linear2"]
    if len(l2):
        l2s = pd.concat([select_by_val(l2[(l2.kind == "fl") & (l2.optimizer == "sgd")], ["fm"], "lr"),
                         l2[(l2.kind == "fl") & (l2.optimizer == "adam")], l2[l2.kind == "central"]])
        summ(l2s, ["fm", "kind", "optimizer", "lr"]).to_csv(OUT / "T_linear2.csv", index=False)

    # 8b. institution clients: FedAvg / FedBN / FedAvgBN with slide-level (micro) and institution-macro accuracy
    fb = df[df.grid.isin(["partition_fedbn", "partition_fedbn_sgd", "partition_bnref"])].copy()
    if len(fb):
        from fedfm.partitions import client_assignment
        inst_of = client_assignment(samples, "institution_all")
        fb["train_macro"] = [r.train_macro if pd.notna(r.train_macro) else train_client_macro_from_pred(r, inst_of) for _, r in fb.iterrows()]
        fbs = pd.concat([select_by_val(fb[fb.optimizer == "sgd"], ["fm", "algorithm"], "lr"), fb[fb.optimizer == "adam"]])
        t = summ(fbs, ["fm", "algorithm", "optimizer", "lr"])
        t["train_macro_sd"] = fbs.groupby(["fm", "algorithm", "optimizer", "lr"]).train_macro.std().values
        t.to_csv(OUT / "T_fedbn_institution.csv", index=False)

    # 9. client selection with coverage and threshold-reaching fractions; two-level CI vs stratified
    selg = df[df.grid == "selection"]
    if len(selg):
        allsel = pd.concat([selg, main[(main.algorithm == "FedAvg") & (main.fm.isin(selg.fm.unique()))]])
        allsel = allsel.drop_duplicates(subset=["fm", "sampling", "seed"], keep="first")
        t = summ(allsel, ["fm", "sampling"])
        rows = []
        for _, r in allsel.iterrows():
            h = r.val_hist if r.val_hist is not None else json.load(open(r.path))["val_history"]
            cov = r.coverage
            rows.append(dict(fm=r.fm, sampling=r.sampling, seed=r.seed,
                             r60=next((i + 1 for i, v in enumerate(h) if v >= 0.6), np.nan), reach60=any(v >= 0.6 for v in h),
                             r80=next((i + 1 for i, v in enumerate(h) if v >= 0.8), np.nan), reach80=any(v >= 0.8 for v in h),
                             coverage_frac=(np.mean([c == 9 for c in cov]) if cov else np.nan),
                             max_sel_share=(max(r.sel_counts.values()) / len(h) if r.sel_counts else np.nan)))
        th = pd.DataFrame(rows).groupby(["fm", "sampling"]).agg(r60=("r60", "mean"), reach60=("reach60", "mean"),
                                                                r80=("r80", "mean"), r80_sd=("r80", "std"), reach80=("reach80", "mean"),
                                                                coverage_frac=("coverage_frac", "mean"), max_sel_share=("max_sel_share", "mean")).reset_index()
        t = t.merge(th, on=["fm", "sampling"]); t.to_csv(OUT / "T_selection.csv", index=False); S["selection"] = t.to_dict("records")
        rows = []
        for fm in selg.fm.unique():
            base = allsel[(allsel.fm == fm) & (allsel.sampling == "stratified")]
            A, Bm, labs = [], [], []
            for pol in ["uniform", "ucb", "pathology_aware"]:
                o = allsel[(allsel.fm == fm) & (allsel.sampling == pol)]
                if len(o) and len(base): A.append(o); Bm.append(base); labs.append(pol)
            if labs:
                r_, _ = diff_table(A, Bm, labs, seed_key=f"sel:{fm}")
                for x in r_: x.update(fm=fm, policy=x.pop("label")); rows.append(x)
        pd.DataFrame(rows).to_csv(OUT / "T_selection_diff.csv", index=False)

    # 10. per-class recall + confusion
    rec_rows, conf = [], {}
    for name, sub in [("FedAvg-adam", main[main.algorithm == "FedAvg"]), ("FedProx-adam", main[main.algorithm == "FedProx"]),
                      ("SCAFFOLD-adam", main[main.algorithm == "SCAFFOLD"]),
                      ("FedAvg-sgd", sel[sel.algorithm == "FedAvg"] if len(sel) else E()), ("SCAFFOLD-sgd", sel[sel.algorithm == "SCAFFOLD"] if len(sel) else E()),
                      ("central-tuned", tuned[tuned.optimizer == "adam"] if len(tuned) else E())]:
        for fm in FMS:
            s = sub[sub.fm == fm]
            if not len(s): continue
            m, sd = mean_recall(s)
            rec_rows.append(dict(setting=name, fm=fm, **{c: m[i] for i, c in enumerate(CLASSES)}, **{c + "_sd": sd[i] for i, c in enumerate(CLASSES)}))
            conf[f"{name}|{fm}"] = np.mean(np.array([c for c in s["confusion"]]), axis=0).tolist()
    pd.DataFrame(rec_rows).to_csv(OUT / "T_per_class_recall.csv", index=False)
    json.dump(conf, open(OUT / "confusion_mean.json", "w"))

    # 11. accuracy vs client size
    cs_rows = []
    for name, sub in [("adam", main), ("sgd", sel if len(sel) else E())]:
        for (fm, algo), s in sub.groupby(["fm", "algorithm"]):
            acc_by_client = defaultdict(list)
            for pc in s["per_client"]:
                if pc is None: continue
                for cid, (c, n) in pc.items():
                    acc_by_client[cid].append(c / n)
            for cid, v in acc_by_client.items():
                cs_rows.append(dict(optimizer=name, fm=fm, algorithm=algo, client=cid, cancer=cid.split("-")[1].split("_")[0],
                                    n_train=train_size[cid], acc=float(np.mean(v))))
    cs = pd.DataFrame(cs_rows)
    if len(cs):
        cs.to_csv(OUT / "T_client_size_raw.csv", index=False)
        cs["size_bin"] = pd.cut(cs.n_train, [0, 14, 29, 59, 10_000], labels=["<15", "15-29", "30-59", ">=60"])
        from scipy.stats import spearmanr
        agg = []
        for (o, fm, algo), s in cs.groupby(["optimizer", "fm", "algorithm"]):
            row = dict(optimizer=o, fm=fm, algorithm=algo, rho=spearmanr(s.n_train, s.acc).statistic, p=spearmanr(s.n_train, s.acc).pvalue)
            for b, ss in s.groupby("size_bin", observed=False):
                row[f"acc_{b}"] = ss.acc.mean(); row[f"n_{b}"] = len(ss)
            agg.append(row)
        pd.DataFrame(agg).to_csv(OUT / "T_client_size.csv", index=False)

    # 12. legacy reproduction
    legacy = df[df.grid == "legacy_droplast"]
    if len(legacy):
        t = summ(pd.concat([legacy, main[main.algorithm.isin(["FedAvg", "SCAFFOLD"])]]), ["fm", "algorithm", "legacy_drop_last"])
        t.to_csv(OUT / "T_legacy_droplast.csv", index=False); S["legacy"] = t.to_dict("records")

    # 13. cost pieces per head / algorithm: bytes to the selected checkpoint and until stopping
    rows = []
    for name, sub, head in [("FedAvg-adam-mlp", main[main.algorithm == "FedAvg"], "mlp"), ("SCAFFOLD-adam-mlp", main[main.algorithm == "SCAFFOLD"], "mlp"),
                            ("FedAvg-sgd-mlp", sel[sel.algorithm == "FedAvg"] if len(sel) else E(), "mlp"),
                            ("FedAvg-adam-linear", lin[lin.kind == "fl"] if len(lin) else E(), "linear")]:
        for fm in FMS:
            s = sub[sub.fm == fm]
            if not len(s): continue
            npar = int(s.n_params.iloc[0]); per_round = npar * 4 * 2 * 20 * (2 if "SCAFFOLD" in name else 1)
            rows.append(dict(setting=name, fm=fm, head=head, n_params=npar, bytes_per_round_total=per_round,
                             best_round=float(s.best_round.mean()), rounds_run=float(s.rounds.mean()),
                             GB_to_best=per_round * s.best_round.mean() / 1e9, GB_to_stop=per_round * s.rounds.mean() / 1e9,
                             seconds=float(s.seconds.mean())))
    pd.DataFrame(rows).to_csv(OUT / "T_cost_head.csv", index=False)

    json.dump(S, open(REV / "analysis" / "summary_cls.json", "w"), indent=1, default=float)
    json.dump(MANIFEST, open(REV / "analysis" / "selection_manifest.json", "w"), indent=1, default=float)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
