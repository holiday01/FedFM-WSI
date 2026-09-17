"""Adapters that reuse the legacy UCB / PathologyAware selectors unchanged."""
import numpy as np
from . import data  # noqa: F401  (puts legacy_copy on sys.path)
from fl_agent.bandit_selection import UCBClientSelector, PathologyAwareSelector  # noqa: E402


def make_selector(kind, clients, X_raw, cfg_ucb_c=2.0):
    ids = [c.cid for c in clients]
    if kind == "ucb":
        return UCBClientSelector(ids, c=cfg_ucb_c)
    if kind == "pathology_aware":
        sel = PathologyAwareSelector(ids, c=cfg_ucb_c, beta=1.0, lam=0.5, rho=0.2, window=10)
        means = []
        for c in clients:                       # first 30 training slides, raw features (as in the legacy code)
            m = X_raw[c.train_idx[:30]].mean(0).cpu().numpy().astype(np.float32)
            sel.feature_means[c.cid] = m
            sel.cancer_map[c.cid] = c.label
            means.append(m)
        sel.global_prototype = np.mean(means, axis=0)
        return sel
    return None
