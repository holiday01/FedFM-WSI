#!/usr/bin/env python
"""
Figures generated from analysis/tables/*.csv.
Every function skips gracefully when its table is missing, so the script can be run on
partial results.  Output: figures/*.pdf (+ .png previews).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REV = Path(__file__).resolve().parents[1]
T = REV / "analysis" / "tables"
FIG = REV / "figures"; FIG.mkdir(exist_ok=True)

# validated palette (dataviz skill): slots 1-4, sequential blue ramp, text inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
LBL = {"UNI_v2": "UNI v2", "Virchow2": "Virchow2", "Phikon_v2": "Phikon v2", "Conch_v15": "CONCH v1.5",
       "CTransPath": "CTransPath", "Midnight12k": "Midnight-12k", "ResNet50": "ResNet50"}
CLASSES = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]
ALGOS = ["FedAvg", "FedProx", "SCAFFOLD", "FedBN"]

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})


def load(name):
    p = T / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def bars(ax, cats, series, width=0.8, ylabel="", ylim=None, legend=True):
    """series: list of (label, mean array, sd array, color)"""
    n = len(series); w = width / n; x = np.arange(len(cats))
    for i, (lab, m, sd, col) in enumerate(series):
        pos = x - width / 2 + w * (i + 0.5)
        ax.bar(pos, m, w * 0.92, color=col, label=lab, linewidth=0)
        if sd is not None:
            ax.errorbar(pos, m, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim[0], ylim[1] + (18 if legend else 0))
    grid(ax)
    if legend: ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8, handlelength=1.2, columnspacing=0.8)


# ─────────────────────── Fig 2: cohort / federation ───────────────────────
def fig2():
    coh = json.load(open(REV / "data" / "cohort_classification.json"))
    st = json.load(open(REV / "analysis" / "cohort_stats.json"))
    inst = load("T_institutions")
    from collections import Counter, defaultdict
    sizes = Counter(s["client_id"] for s in coh)
    cancer_of = {s["client_id"]: CLASSES[s["label"]] for s in coh}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    order = sorted(sizes, key=lambda c: (CLASSES.index(cancer_of[c]), -sizes[c]))
    xs = np.arange(len(order)); vals = [sizes[c] for c in order]
    ax.bar(xs, vals, color=C[0], width=0.85, linewidth=0)
    # cancer-type separators and labels
    pos = 0; ticks, tlabs = [], []
    for cl in CLASSES:
        n = sum(cancer_of[c] == cl for c in order)
        if n == 0: continue
        ticks.append(pos + n / 2 - 0.5); tlabs.append(f"{cl} ({n})")
        if pos > 0: ax.axvline(pos - 0.5, color=GRID, linewidth=0.6)
        pos += n
    ax.set_xticks(ticks); ax.set_xticklabels(tlabs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("107 Project_TSS clients (cancer type, number of clients)")
    ax.set_ylabel("Slides per client"); ax.set_ylim(0, max(vals) * 1.08); grid(ax)
    panel(ax, "A")
    ax = axes[1]
    # institution × cancer heatmap for multi-cancer institutions
    m = defaultdict(Counter)
    tss_inst = {}
    sys.path.insert(0, str(REV / "code"))
    from fedfm.data import tss_institution_map
    imap = tss_institution_map()
    for s in coh:
        m[imap.get(s["tss"], s["tss"])][CLASSES[s["label"]]] += 1
    multi = [i for i in m if len(m[i]) > 1]
    multi = sorted(multi, key=lambda i: -sum(m[i].values()))
    M = np.array([[m[i][c] for c in CLASSES] for i in multi], dtype=float)
    im = ax.imshow(np.log1p(M), cmap=SEQ, aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(multi))); ax.set_yticklabels([i[:22] for i in multi], fontsize=8)
    for r in range(M.shape[0]):
        for c in range(9):
            if M[r, c] > 0:
                ax.text(c, r, int(M[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if M[r, c] > 60 else INK)
    ax.set_title(f"{len(multi)} of {st['n_institutions']} source organisations\nhold > 1 cancer type", fontsize=8.5, loc="right")
    ax.tick_params(length=0)
    panel(ax, "B")
    fig.tight_layout(w_pad=2)
    save(fig, "fig2_federation")


# ─────────────────────── Fig 3: encoders ───────────────────────
def fig3():
    adam, sgd, cen, gap = load("T_main_adam"), load("T_main_sgd"), load("T_central"), load("T_fl_gap")
    if adam is None: return
    fa = adam[adam.algorithm == "FedAvg"].set_index("fm")
    fs = sgd[sgd.algorithm == "FedAvg"].set_index("fm") if sgd is not None else None
    ca = cen[(cen.protocol == "tuned") & (cen.optimizer == "adam")].set_index("fm") if cen is not None else None
    cs_ = cen[(cen.protocol == "tuned") & (cen.optimizer == "sgd")].set_index("fm") if cen is not None else None
    fms = [f for f in FMS if f in fa.index]
    cats = [LBL[f] for f in fms]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1))
    for k, (metric, name) in enumerate([("acc", "Accuracy (%)"), ("bacc", "Balanced accuracy (%)")]):
        ser = [("FL, Adam (fixed)", 100 * fa.loc[fms, metric].values, 100 * fa.loc[fms, metric + "_sd"].values, C[0])]
        if fs is not None and all(f in fs.index for f in fms):
            ser.append(("FL, SGD", 100 * fs.loc[fms, metric].values, 100 * fs.loc[fms, metric + "_sd"].values, C[1]))
        if ca is not None and all(f in ca.index for f in fms):
            ser.append(("Central, Adam", 100 * ca.loc[fms, metric].values, 100 * ca.loc[fms, metric + "_sd"].values, C[2]))
        if cs_ is not None and all(f in cs_.index for f in fms):
            ser.append(("Central, SGD", 100 * cs_.loc[fms, metric].values, 100 * cs_.loc[fms, metric + "_sd"].values, C[3]))
        bars(axes[k], cats, ser, ylabel=name, ylim=(0, 105), legend=False)
        panel(axes[k], "AB"[k])
    h_, l_ = axes[0].get_legend_handles_labels()
    axes[0].legend(h_, l_, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.05, -0.42), fontsize=7.5, handlelength=1.2, columnspacing=0.8)
    ax = axes[2]
    if gap is not None and len(gap):
        y = np.arange(len(fms))
        for i, (opt, col, off, lab) in enumerate([("adam", C[0], -0.22, "FL Adam (fixed) vs central Adam"), ("adam_sel", C[3], 0.0, "FL Adam (selected) vs central Adam"), ("sgd", C[1], 0.22, "FL SGD vs central SGD")]):
            g = gap[(gap.optimizer == opt) & (gap.fm != "MEAN_PATHOLOGY")].set_index("fm")
            if not len(g): continue
            gg = g.reindex(fms)
            ax.errorbar(100 * gg["diff"], y + off, xerr=[100 * (gg["diff"] - gg.lo), 100 * (gg.hi - gg["diff"])],
                        fmt="o", color=col, ecolor=col, ms=4, capsize=2, elinewidth=1, label=lab)
        ax.set_yticks(y); ax.set_yticklabels(cats); ax.invert_yaxis()
        ax.axvline(0, color=INK2, linewidth=0.6)
        ax.set_xlabel("Centralized − FL (pp)"); grid(ax, "x")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.3, -0.28), ncol=1, fontsize=8)
    panel(ax, "C")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_fm_comparison")


# ─────────────────────── Fig 4: algorithms × optimiser ───────────────────────
def fig4():
    adam, sgd, diff = load("T_main_adam"), load("T_main_sgd"), load("T_algo_vs_fedavg")
    if adam is None: return
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.1]))
    for k, (name, t) in enumerate([("Adam (fixed lr 3e-4)", adam), ("SGD (lr selected on validation)", sgd)]):
        ax = axes[k]
        if t is None: panel(ax, "AB"[k]); continue
        fms = [f for f in FMS if f in set(t.fm)]
        ser = []
        for i, a in enumerate(ALGOS):
            s = t[t.algorithm == a].set_index("fm").reindex(fms)
            ser.append((a, 100 * s.acc.values, 100 * s.acc_sd.values, C[i]))
        bars(ax, [LBL[f] for f in fms], ser, ylabel="Accuracy (%)" if k == 0 else "", ylim=(0, 105), legend=(k == 0))
        ax.set_title(name, fontsize=9); panel(ax, "AB"[k])
    ax = axes[2]
    if diff is not None and len(diff):
        yl = []; y = 0
        for a in ["FedProx", "SCAFFOLD", "FedBN"]:
            for o, col, oname in [("adam", C[0], "Adam"), ("adam_sel", C[3], "Adam sel."), ("sgd", C[1], "SGD")]:
                m = diff[(diff.algorithm == a) & (diff.optimizer == o) & (diff.fm == "MEAN_PATHOLOGY")]
                d = diff[(diff.algorithm == a) & (diff.optimizer == o) & (diff.fm != "MEAN_PATHOLOGY") & (diff.fm != "ResNet50")]
                if not len(m): continue
                m = m.iloc[0]
                ax.errorbar(100 * m["diff"], y, xerr=[[100 * (m["diff"] - m.lo)], [100 * (m.hi - m["diff"])]], fmt="o", color=col, ms=4, capsize=2, elinewidth=1)
                ax.scatter(100 * d["diff"], np.full(len(d), y) + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)), s=8, color=col, alpha=0.5, linewidths=0)
                yl.append(f"{a}, {oname}"); y += 1
        ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color=INK2, linewidth=0.6); grid(ax, "x")
        ax.set_xlabel("Δ vs FedAvg (pp)")
    panel(ax, "C")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_algorithms")


# ─────────────────────── Fig 5: per-class recall + client size ───────────────────────
def fig5():
    rec, cs = load("T_per_class_recall"), load("T_client_size")
    if rec is None: return
    settings = [("FedAvg-adam", "FL, Adam"), ("FedAvg-sgd", "FL, SGD"), ("central-tuned", "Centralized")]
    settings = [s for s in settings if s[0] in set(rec.setting)]
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    for k, (key, name) in enumerate(settings):
        ax = fig.add_subplot(gs[0, k])
        r = rec[rec.setting == key].set_index("fm").reindex([f for f in FMS if f in set(rec.fm)])
        M = (100 * r[CLASSES].values).T                                   # classes x encoders
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 60 else INK)
        ax.set_xticks(range(len(r))); ax.set_xticklabels([LBL[f] for f in r.index], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(9)); ax.set_yticklabels(CLASSES if k == 0 else [], fontsize=8)
        ax.set_title(name, fontsize=9); ax.tick_params(length=0)
        panel(ax, "ABC"[k])
    if cs is not None and len(cs):
        binsc = ["<15", "15-29", "30-59", ">=60"]
        for k, opt in enumerate(["adam", "sgd"]):
            ax = fig.add_subplot(gs[1, k])
            d = cs[(cs.optimizer == opt) & (cs.fm != "ResNet50")]
            if not len(d): continue
            for i, a in enumerate(ALGOS):
                dd = d[d.algorithm == a]
                if not len(dd): continue
                m = [100 * dd[f"acc_{b}"].mean() for b in binsc]
                sd = [100 * dd[f"acc_{b}"].std() for b in binsc]
                ax.errorbar(range(4), m, yerr=sd, color=C[i], marker="o", ms=4, capsize=2, linewidth=1.5, label=a)
            ax.set_xticks(range(4)); ax.set_xticklabels([f"{b}" for b in binsc]); ax.set_ylim(0, 105)
            ax.set_xlabel("Client training slides"); ax.set_ylabel("Per-client accuracy (%)" if k == 0 else "")
            ax.set_title("Adam" if opt == "adam" else "SGD", fontsize=9); grid(ax)
            if k == 0: ax.legend(frameon=False, fontsize=7.5)
            panel(ax, "DE"[k])
        ax = fig.add_subplot(gs[1, 2])
        d = cs[cs.fm != "ResNet50"]
        MK = {"UNI_v2": "o", "Virchow2": "s", "Phikon_v2": "^", "Conch_v15": "D", "CTransPath": "v", "Midnight12k": "P"}
        for i, opt in enumerate(["adam", "sgd"]):
            for fm, mk in MK.items():
                dd = d[(d.optimizer == opt) & (d.fm == fm)]
                if len(dd):
                    ax.scatter(dd.rho, [ALGOS.index(a) + (i - 0.5) * 0.3 for a in dd.algorithm], s=18, color=C[i], marker=mk,
                               alpha=0.85, linewidths=0, label=LBL[fm] if i == 0 else None)
        ax.set_yticks(range(4)); ax.set_yticklabels(ALGOS); ax.invert_yaxis(); ax.axvline(0, color=INK2, linewidth=0.6)
        ax.set_xlabel("Spearman ρ (blue Adam, orange SGD)"); grid(ax, "x")
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), handletextpad=0.2, columnspacing=0.8)
        panel(ax, "F")
    fig.tight_layout(h_pad=2, w_pad=1.2)
    save(fig, "fig5_per_class_clientsize")


# ─────────────────────── Fig 6: survival ───────────────────────
def fig6():
    P = load("T_surv_paired")
    if P is None or not len(P): return
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    for k, cancer in enumerate(["BRCA", "COAD", "STAD"]):
        ax = fig.add_subplot(gs[0, k])
        d = P[P.cancer == cancer].set_index("fm")
        fms = [f for f in FMS if f in d.index]; y = np.arange(len(fms)); d = d.reindex(fms)
        for i, (c, lab, lo, hi, off) in enumerate([
                ("fl", "Federated", "fl_lo", "fl_hi", -0.25), ("central", "Centralized (pooled)", "central_lo", "central_hi", 0),
                ("strat", "Centralized (site-stratified)", "strat_lo", "strat_hi", 0.25)]):
            if c not in d.columns: continue
            ax.errorbar(d[c], y + off, xerr=[d[c] - d[lo], d[hi] - d[c]], fmt="o", ms=3.5, color=C[i], ecolor=C[i],
                        elinewidth=0.9, capsize=1.5, label=lab)
        ax.axvline(0.5, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_yticks(y); ax.set_yticklabels([LBL[f] for f in fms] if k == 0 else []); ax.invert_yaxis()
        ax.set_xlim(0.35, 0.85); ax.set_xlabel("Patient-level C-index" if k == 1 else ""); ax.set_title(cancer, fontsize=9); grid(ax, "x")
        if k == 1: ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
        panel(ax, "ABC"[k])
    ax = fig.add_subplot(gs[1, :])
    P2 = P.copy(); P2["lab"] = P2.fm.map(LBL)
    P2 = P2.sort_values(["cancer", "diff"], ascending=[True, False]).reset_index(drop=True)
    x = np.arange(len(P2))
    cols = [C[0] if lo > 0 else (C[1] if hi < 0 else INK2) for lo, hi in zip(P2.diff_lo, P2.diff_hi)]
    for xi, (d0, lo, hi, col) in enumerate(zip(P2["diff"], P2.diff_lo, P2.diff_hi, cols)):
        ax.plot([xi, xi], [lo, hi], color=col, linewidth=1.0, solid_capstyle="butt")
    ax.scatter(x, P2["diff"], s=14, color=cols, linewidths=0, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.6)
    for cancer in ["BRCA", "COAD", "STAD"]:
        idx = np.where(P2.cancer == cancer)[0]
        ax.text(idx.mean(), 0.27, cancer, ha="center", va="top", fontsize=8, color=INK2)
        if idx[0] > 0: ax.axvline(idx[0] - 0.5, color=GRID, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(P2.lab, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("FL − centralized (pooled)\nC-index, 95% CI"); ax.set_ylim(-0.3, 0.3); grid(ax)
    ax.text(-0.045, 1.06, "D", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    fig.tight_layout(h_pad=2.5)
    save(fig, "fig6_survival")


# ─────────────────────── Fig 7: selection + sweeps ───────────────────────
def fig7():
    sel, mu, lr, part = load("T_selection"), load("T_mu_adam"), load("T_lr_adam"), load("T_participation")
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 3.0))
    ax = axes[0]
    if sel is not None and len(sel):
        # raw seeds for the strip
        import glob
        rows = []
        for f in glob.glob(str(REV / "results" / "cls" / "selection" / "UNI_v2" / "*.json")):
            j = json.load(open(f)); c = j["config"]
            if c["algorithm"] == "FedAvg" and c["optimizer"] == "adam":
                rows.append((c["sampling"], j["test"]["accuracy"], j["rounds_run"]))
        R = pd.DataFrame(rows, columns=["sampling", "acc", "rounds"])
        pols = [p for p in ["stratified", "uniform", "ucb", "pathology_aware"] if p in set(R.sampling)]
        names = {"stratified": "Stratified", "uniform": "Uniform", "ucb": "UCB1", "pathology_aware": "PathologyAware"}
        selt = load("T_selection")
        for i, p in enumerate(pols):
            v = 100 * R[R.sampling == p].acc.values
            ax.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.15, 0.15, len(v)), v, s=10, color=C[0], alpha=0.6, linewidths=0)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="_", color=INK, ms=14, capsize=4, elinewidth=1.2)
            reach = selt[(selt.fm == "UNI_v2") & (selt.sampling == p)].reach80 if selt is not None and "reach80" in selt.columns else None
            lab = (f"{int(round(float(reach.iloc[0]) * len(v)))}/{len(v)}" if reach is not None and len(reach) else f"n={len(v)}")
            ax.text(i, 103.5, lab, ha="center", va="top", fontsize=7, color=INK2)
        ax.set_ylim(55, 105)
        ax.set_xticks(range(len(pols))); ax.set_xticklabels([names[p] for p in pols], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Test accuracy (%), UNI v2"); grid(ax)
    panel(ax, "A")
    for k, (t, hp, name, log) in enumerate([(mu, "mu", "FedProx μ", True), (lr, "lr", "Adam learning rate", True), (part, "k", "Clients per round", False)]):
        ax = axes[k + 1]
        if t is None or not len(t): panel(ax, "BCD"[k]); continue
        d = t[t.fm != "ResNet50"].copy()
        if hp == "k": d["k"] = d["k"].replace(-1, 107)
        g = d.groupby(hp).agg(val=("val", "mean"), acc=("acc", "mean"), acc_sd=("acc", "std")).reset_index()
        x = np.log10(g[hp]) if log else np.arange(len(g))
        ax.plot(x, 100 * g.val, "-o", color=C[0], ms=4, label="validation (selection)")
        ax.errorbar(x, 100 * g.acc, yerr=100 * g.acc_sd, fmt="-s", color=C[1], ms=4, capsize=2, label="test")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in g[hp]], fontsize=8)
        ax.set_xlabel(name); ax.set_ylim(40, 100); grid(ax)
        if k == 0: ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_ylabel("Accuracy (%), mean over FMs" if k == 0 else "")
        panel(ax, "BCD"[k])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig7_selection_sweeps")


# ─────────────────────── Fig 8: partitions ───────────────────────
def fig8():
    t = load("T_partition")
    if t is None or not len(t): return
    parts = ["project_tss", "institution", "labelskew_0.1", "labelskew_1.0", "iid"]
    names = {"project_tss": "Project_TSS\n(1 class/client)", "institution": "Institution\n(58 clients)",
             "labelskew_0.1": "Label skew\nα=0.1", "labelskew_1.0": "Label skew\nα=1.0", "iid": "IID"}
    algos = [a for a in ["FedAvg", "FedProx"] if a in set(t.algorithm)]
    fig, axes = plt.subplots(1, len(algos), figsize=(3.6 * len(algos), 3.0), squeeze=False)
    for k, a in enumerate(algos):
        ax = axes[0, k]
        d = t[t.algorithm == a]
        fms = [f for f in ["UNI_v2", "Conch_v15", "Phikon_v2", "Virchow2"] if f in set(d.fm)]
        ps = [p for p in parts if p in set(d.partition)]
        for i, fm in enumerate(fms):
            dd = d[d.fm == fm].set_index("partition").reindex(ps)
            ax.errorbar(np.arange(len(ps)) + (i - (len(fms) - 1) / 2) * 0.12, 100 * dd.acc, yerr=100 * dd.acc_sd,
                        fmt="o", color=C[i], ms=4, capsize=2, label=LBL[fm])
        ax.set_xticks(range(len(ps))); ax.set_xticklabels([names[p] for p in ps], fontsize=7.5)
        ax.set_ylabel("Test accuracy (%)" if k == 0 else ""); ax.set_ylim(40, 100); ax.set_title(a, fontsize=9); grid(ax)
        if k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        panel(ax, "AB"[k])
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_partitions")


# ─────────────────────── Fig 9: external CPTAC ───────────────────────
def fig9():
    p = REV / "results" / "external" / "external_eval.csv"
    if not p.exists(): return
    E = pd.read_csv(p)
    sgd = load("T_main_sgd")
    sets = [("main", "FedAvg", "adam", None, "FL, Adam (fixed)"), ("sgd", "FedAvg", "sgd", sgd, "FL, SGD"), ("central", "central-tuned", "adam", None, "Central, Adam"), ("central", "central-tuned", "sgd", None, "Central, SGD")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, coh in enumerate(["luad", "pda"]):
        ax = axes[k]
        ser = []
        fms = [f for f in FMS if f in set(E[E.cohort == coh].fm)]
        for i, (grid_, algo, opt, seltab, name) in enumerate(sets):
            d = E[(E.grid == grid_) & (E.algorithm == algo) & (E.cohort == coh) & (E.optimizer == opt)]
            if grid_ == "sgd" and seltab is not None:       # keep validation-selected lr per FM
                lrs = seltab[seltab.algorithm == "FedAvg"].set_index("fm").lr
                d = d[[abs(r.lr - lrs.get(r.fm, -1)) < 1e-9 for _, r in d.iterrows()]]
            if grid_ == "central":
                cen = load("T_central"); lrs = cen[(cen.protocol == "tuned") & (cen.optimizer == opt)].set_index("fm").lr
                d = d[[abs(r.lr - lrs.get(r.fm, -1)) < 1e-9 for _, r in d.iterrows()]]
            if not len(d): continue
            g = d.groupby("fm").agg(r=("recall", "mean"), sd=("recall", "std"), t=("tcga_recall", "mean")).reindex(fms)
            ser.append((name, 100 * g.r.values, 100 * g.sd.values, C[i]))
        if ser:
            bars(ax, [LBL[f] for f in fms], ser, ylabel="Recall on CPTAC (%)" if k == 0 else "", ylim=(0, 112), legend=(k == 0))
        ax.set_title({"luad": "CPTAC-LUAD (244 slides) → LUAD class", "pda": "CPTAC-PDA (169 slides) → PAAD class"}[coh], fontsize=8.5)
        panel(ax, "AB"[k])
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig9_external")


# ─────────────────────── Additional figures ───────────────────────
def figS_convergence():
    import glob
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for k, (grid_, opt, name) in enumerate([("main", "adam", "Adam, lr 3e-4"), ("sgd", "sgd", "SGD")]):
        ax = axes[k]
        sgd = load("T_main_sgd")
        for i, a in enumerate(ALGOS):
            H = []
            for f in glob.glob(str(REV / "results" / "cls" / grid_ / "UNI_v2" / "*.json")):
                j = json.load(open(f)); c = j["config"]
                if c["algorithm"] != a: continue
                if grid_ == "sgd" and sgd is not None:
                    lr = sgd[(sgd.fm == "UNI_v2") & (sgd.algorithm == a)].lr
                    if not len(lr) or abs(c["lr"] - lr.iloc[0]) > 1e-9: continue
                H.append(j["val_history"])
            if not H: continue
            L = min(len(h) for h in H); M = np.array([h[:L] for h in H])
            x = np.arange(1, L + 1)
            ax.plot(x, 100 * M.mean(0), color=C[i], linewidth=1.4, label=a)
            ax.fill_between(x, 100 * (M.mean(0) - M.std(0)), 100 * (M.mean(0) + M.std(0)), color=C[i], alpha=0.15, linewidth=0)
        ax.set_xlabel("Communication round"); ax.set_ylabel("Validation accuracy (%)" if k == 0 else ""); ax.set_ylim(0, 100)
        ax.set_title(f"UNI v2, {name}", fontsize=9); grid(ax)
        if k == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        panel(ax, "AB"[k])
    fig.tight_layout(w_pad=1.5)
    save(fig, "figS_convergence")


def figS_controls():
    t, lin = load("T_controls"), load("T_linear_vs_mlp")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    if t is not None and len(t):
        for k, kind in enumerate(["fl", "central"]):
            ax = axes[k]
            d = t[t.kind == kind]
            fms = [f for f in FMS if f in set(d.fm)]
            ser = []
            for i, (ft, name) in enumerate([("raw", "raw features"), ("zscore", "standardised"), ("zscore_rp", "standardised + RP 6,144")]):
                dd = d[d.ft == ft].set_index("fm").reindex(fms)
                if dd.acc.isna().all(): continue
                ser.append((name, 100 * dd.acc.values, 100 * dd.acc_sd.values, C[i]))
            bars(ax, [LBL[f] for f in fms], ser, ylabel="Accuracy (%)" if k == 0 else "", ylim=(0, 105), legend=(k == 0))
            ax.set_title("FedAvg" if kind == "fl" else "Centralized", fontsize=9); panel(ax, "AB"[k])
    ax = axes[2]
    lin2 = load("T_linear2"); adam = load("T_main_adam"); c2 = load("T_controls2")
    if lin is not None and lin2 is not None and adam is not None and c2 is not None:
        fms = [f for f in FMS if f in set(lin.fm)]
        ser = [("MLP, raw, Adam", *[100 * adam[adam.algorithm == "FedAvg"].set_index("fm").reindex(fms)[c].values for c in ("acc", "acc_sd")], C[0]),
               ("linear, raw, Adam", *[100 * lin[(lin.kind == "fl") & (lin.arch == "linear")].set_index("fm").reindex(fms)[c].values for c in ("acc", "acc_sd")], C[1]),
               ("MLP, standardised, SGD", *[100 * c2[(c2.algorithm == "FedAvg") & (c2.optimizer == "sgd")].set_index("fm").reindex(fms)[c].values for c in ("acc", "acc_sd")], C[2]),
               ("linear, standardised, SGD", *[100 * lin2[(lin2.kind == "fl") & (lin2.optimizer == "sgd")].set_index("fm").reindex(fms)[c].values for c in ("acc", "acc_sd")], C[3])]
        bars(ax, [LBL[f] for f in fms], ser, ylabel="", ylim=(0, 105))
        ax.set_title("FedAvg: head and features", fontsize=9)
    panel(ax, "C")
    fig.tight_layout(w_pad=1.2, h_pad=2.5)
    save(fig, "figS_controls_linear")


def figS_confusion():
    p = T / "confusion_mean.json"
    if not p.exists(): return
    conf = json.load(open(p))
    keys = [k for k in ["FedAvg-adam|UNI_v2", "FedAvg-sgd|UNI_v2", "SCAFFOLD-adam|UNI_v2", "SCAFFOLD-sgd|UNI_v2", "central-tuned|UNI_v2"] if k in conf]
    if not keys: return
    ncol = 3; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.6 * nrow), squeeze=False)
    for k, key in enumerate(keys):
        ax = axes[k // ncol, k % ncol]; M = np.array(conf[key]); Mn = 100 * M / M.sum(1, keepdims=True)
        ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=100)
        for i in range(9):
            for j in range(9):
                if Mn[i, j] >= 5: ax.text(j, i, f"{Mn[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if Mn[i, j] > 60 else INK)
        ax.set_xticks(range(9)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=8); ax.set_yticks(range(9))
        ax.set_yticklabels(CLASSES if k % ncol == 0 else [], fontsize=8); ax.set_title(key.split("|")[0].replace("-", ", "), fontsize=9)
        ax.tick_params(length=0)
        if k % ncol == 0: ax.set_ylabel("True class")
        ax.set_xlabel("Predicted")
        panel(ax, "ABCDE"[k])
    for k in range(len(keys), nrow * ncol): axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout(w_pad=0.8, h_pad=1.5)
    save(fig, "figS_confusion")


def figS_legacy():
    t = load("T_legacy_droplast")
    if t is None or not len(t): return
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    fms = [f for f in FMS if f in set(t.fm)]
    ser = []
    for i, (a, leg, name) in enumerate([("FedAvg", True, "FedAvg, legacy batching"), ("FedAvg", False, "FedAvg, v2 batching"),
                                        ("SCAFFOLD", True, "SCAFFOLD, legacy batching"), ("SCAFFOLD", False, "SCAFFOLD, v2 batching")]):
        d = t[(t.algorithm == a) & (t.legacy_drop_last == leg)].set_index("fm").reindex(fms)
        if d.acc.isna().all(): continue
        ser.append((name, 100 * d.acc.values, 100 * d.acc_sd.values, C[i]))
    bars(ax, [LBL[f] for f in fms], ser, ylabel="Accuracy (%)", ylim=(0, 105))
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "figS_legacy_batching")


def figS_clientsize_fm():
    cs = load("T_client_size")
    if cs is None or not len(cs): return
    binsc = ["<15", "15-29", "30-59", ">=60"]
    fms = [f for f in FMS if f in set(cs.fm)]
    fig, axes = plt.subplots(2, len(fms), figsize=(1.6 * len(fms), 4.4), sharey=True, squeeze=False)
    for r_, opt in enumerate(["adam", "sgd"]):
        for c_, fm in enumerate(fms):
            ax = axes[r_, c_]
            d = cs[(cs.optimizer == opt) & (cs.fm == fm)]
            for i, a in enumerate(ALGOS):
                dd = d[d.algorithm == a]
                if not len(dd): continue
                ax.plot(range(4), [100 * dd[f"acc_{b}"].iloc[0] for b in binsc], color=C[i], marker="o", ms=3.5, linewidth=1.3, label=a)
            ax.set_xticks(range(4)); ax.set_xticklabels(binsc, rotation=60, fontsize=8); ax.set_ylim(0, 105); grid(ax)
            if r_ == 0: ax.set_title(LBL[fm], fontsize=9)
            if c_ == 0: ax.set_ylabel(("Adam" if opt == "adam" else "SGD") + "\nper-client accuracy (%)")
            if r_ == 0 and c_ == 0: ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.supxlabel("Client training slides", fontsize=10)
    fig.tight_layout()
    save(fig, "figS_clientsize_fm")


def fig1():
    """Schematic overview."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
    def box(x, y, w, h, title, lines, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", linewidth=1.0, edgecolor=col, facecolor=col + "14"))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.8, lines, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)
    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=INK2))
    for x, letter in [(0.5, "A"), (34.5, "B"), (68.5, "C")]:
        ax.text(x, 47.5, letter, fontsize=13, fontweight="bold", va="top")
    box(1, 25, 31, 21, "Slides and encoders", "9 cancer types, 5,208 slides,\n3,707 patients (TCGA)\n6 pathology FMs + ResNet50\ntiles at 20×, 256 px, frozen", C[0])
    box(1, 1, 31, 21, "Slide vectors", "8 tile statistics per dim.\n(mean, s.d., max, min,\np25, p50, p75, p90)\n6,144 to 20,480 dims, cached", C[0])
    box(35, 25, 31, 21, "Federation", "107 Project_TSS clients,\none cancer type each\npatient-level 70/10/20 split\n+ institution, label-skew,\nIID partitions", C[1])
    box(35, 1, 31, 21, "Heads and algorithms", "MLP or linear head\n20 clients per round\nFedAvg / FedProx /\nSCAFFOLD / FedBN\nAdam (fixed / selected η)\nor SGD (selected η)", C[1])
    box(69, 25, 30, 21, "Evaluation", "9-class accuracy, balanced\naccuracy, per-class recall\n5 seeds (10: UNI v2 selection)\nseed + patient bootstrap CIs\nmatched centralized baselines", C[2])
    box(69, 1, 30, 21, "Transfer and survival", "external CPTAC slides\n(LUAD, PDA)\nwithin-cancer Cox\n(BRCA, COAD, STAD)\nvs pooled and site-stratified\ncentralized Cox", C[2])
    arrow(32.3, 34.7, 35.5); arrow(66.3, 68.7, 35.5)
    arrow(32.3, 34.7, 11.5); arrow(66.3, 68.7, 11.5)
    save(fig, "fig1_pipeline")


if __name__ == "__main__":
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, figS_convergence, figS_controls, figS_confusion, figS_legacy, figS_clientsize_fm]:
        try:
            fn()
        except Exception as e:
            import traceback; print("FAILED", fn.__name__, e); traceback.print_exc()
