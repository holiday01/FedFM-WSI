"""Classification / survival metrics and resampling-based uncertainty."""
from collections import defaultdict

import numpy as np

NUM_CLASSES = 9


def classification_metrics(y_true, y_pred, client_ids=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    support = cm.sum(1)
    recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    precision_den = cm.sum(0)
    precision = np.where(precision_den > 0, np.diag(cm) / np.maximum(precision_den, 1), 0.0)
    f1 = np.where(precision + np.nan_to_num(recall) > 0,
                  2 * precision * np.nan_to_num(recall) / np.maximum(precision + np.nan_to_num(recall), 1e-12), 0.0)
    present = support > 0
    out = dict(
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float(np.nanmean(recall[present])),
        macro_f1=float(f1[present].mean()),
        per_class_recall=[None if np.isnan(r) else float(r) for r in recall],
        confusion=cm.tolist(),
        n=int(len(y_true)),
    )
    if client_ids is not None:
        per = defaultdict(lambda: [0, 0])
        for c, t, p in zip(client_ids, y_true, y_pred):
            per[c][0] += int(t == p)
            per[c][1] += 1
        out["per_client"] = {c: [v[0], v[1]] for c, v in per.items()}
        out["per_client_macro_accuracy"] = float(np.mean([v[0] / v[1] for v in per.values()]))
    return out


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    rec = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(rec))


# ───────────────────────────── C-index ─────────────────────────────

def harrell_c(risk, time, event):
    """Harrell's C with the lifelines convention (lifelines.utils.concordance_index):
    a pair is comparable if the earlier time is an event, or if the times are tied and exactly
    one member is an event (the event is treated as earlier); tied event times are not
    comparable; tied predicted risks count one half.  Vectorised O(n^2); verified equal to
    lifelines on data with tied times and tied risks (analysis/verify.py)."""
    risk = np.asarray(risk, float); time = np.asarray(time, float); event = np.asarray(event).astype(bool)
    if event.sum() == 0:
        return np.nan
    ti, tj = time[:, None], time[None, :]
    ei, ej = event[:, None], event[None, :]
    comp = ei & ((ti < tj) | ((ti == tj) & ~ej))          # i is the earlier event of the pair
    ri, rj = risk[:, None], risk[None, :]
    conc = comp & (ri > rj)
    tie = comp & (ri == rj)
    den = comp.sum()
    return float((conc.sum() + 0.5 * tie.sum()) / den) if den > 0 else np.nan


def cluster_bootstrap_indices(groups, B, rng):
    """Yield index arrays resampling whole groups (patients) with replacement."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])
