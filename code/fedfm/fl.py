"""
Federated classification engine (Protocol v2).

Algorithmic semantics follow the legacy implementation
(legacy_copy/fl_agent/federated_learning.py); every intentional change is flagged
with `# [v2]`.  The most important are:

  [v2-1] Checkpoint selection / early stopping use ONLY the pooled validation split.
         The held-out test split is touched exactly once, after training.
  [v2-2] Local mini-batching no longer drops the last incomplete batch
         (legacy: drop_last=True, so any client with <32 training slides performed
         zero local steps).  `legacy_drop_last=True` reproduces the old behaviour.
  [v2-3] Explicit seeds for initialisation, client sampling, shuffling and dropout.
  [v2-4] FedBN: client BN states are snapshotted together with the best global
         weights; monitoring uses macro per-client validation accuracy over all clients.
  [v2-5] Optional SGD local optimiser (SCAFFOLD control experiment), optional
         sample-weighted aggregation / uniform sampling (multi-class partitions),
         optional feature standardisation and parameter-matched random projection.
"""
import copy
import math
import time
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .models import MLPClassifier, LinearProbe, n_params
from .metrics import classification_metrics

NUM_CLASSES = 9


@dataclass
class FLConfig:
    fm: str = "UNI_v2"
    algorithm: str = "FedAvg"            # FedAvg | FedProx | SCAFFOLD | FedBN
    optimizer: str = "adam"              # adam | sgd
    lr: float = 3e-4
    momentum: float = 0.0
    mu: float = 0.01
    local_epochs: int = 1
    batch_size: int = 32
    clients_per_round: Optional[int] = 20
    sampling: str = "stratified"         # stratified | uniform | ucb | pathology_aware
    aggregation: str = "class_balanced"  # class_balanced | sample_weighted
    legacy_drop_last: bool = False
    max_rounds: int = 1000
    patience: int = 50
    min_delta: float = 0.001
    min_rounds: int = 100
    fixed_rounds: Optional[int] = None   # run exactly this many rounds (still val-selected)
    arch: str = "mlp"                    # mlp | linear
    hidden_dims: tuple = (512, 256)
    dropout: float = 0.3
    class_weighting: bool = True
    feature_transform: str = "raw"       # raw | zscore | zscore_rp
    rp_dim: int = 6144
    partition: str = "project_tss"
    seed: int = 0
    tag: str = ""


# ───────────────────────────── helpers ─────────────────────────────

def _state_zero_like(state):
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()}


def _add_scaled_(acc, state, w):
    for k, v in state.items():
        acc[k].add_(v.float(), alpha=w)


class Client:
    def __init__(self, cid, train_idx, val_idx, test_idx, label):
        self.cid = cid
        self.client_id = cid                 # selector compatibility
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.n_train = int(train_idx.numel())
        self.label = label
        self.local_bn = None
        self.c_i = None


def build_clients(samples, client_of, device):
    """client_of: list (aligned with samples) giving the client id of every TRAIN/VAL
    sample (test samples keep their Project_TSS client for per-client reporting)."""
    groups = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for i, s in enumerate(samples):
        groups[client_of[i]][s["split"]].append(i)
    clients = []
    for cid in sorted(groups):
        g = groups[cid]
        if not g["train"]:
            continue
        labels = [samples[i]["label"] for i in g["train"]]
        if cid.startswith("TCGA-"):                  # legacy rule: label of first train sample
            lab = labels[0]
        else:
            lab = int(np.bincount(labels, minlength=NUM_CLASSES).argmax())
        t = lambda l: torch.tensor(l, dtype=torch.long, device=device)
        clients.append(Client(cid, t(g["train"]), t(g["val"]), t(g["test"]), lab))
    return clients


def make_model(cfg, input_dim, batch_norm):
    if cfg.arch == "linear":
        return LinearProbe(input_dim, NUM_CLASSES)
    return MLPClassifier(input_dim, NUM_CLASSES, cfg.hidden_dims, cfg.dropout, batch_norm)


def transform_features(X, train_mask, cfg, device):
    """[v2-5] feature normalisation / parameter-matched projection (FL-realisable:
    per-dimension mean/variance are sample-weighted averages of client statistics;
    the projection matrix is data-independent and shared through a common seed)."""
    if cfg.feature_transform == "raw":
        return X
    Xt = X[train_mask]
    mu = Xt.mean(0, keepdim=True)
    sd = Xt.std(0, keepdim=True)
    Z = (X - mu) / (sd + 1e-6)
    if cfg.feature_transform == "zscore":
        return Z
    if cfg.feature_transform == "zscore_rp":
        g = torch.Generator(device="cpu").manual_seed(12345)
        D = Z.shape[1]
        out = torch.empty((Z.shape[0], cfg.rp_dim), device=device)
        R = (torch.randn(cfg.rp_dim, D, generator=g) / math.sqrt(cfg.rp_dim)).to(device)
        for s in range(0, Z.shape[0], 512):
            out[s:s + 512] = Z[s:s + 512] @ R.T
        del R
        return out
    raise ValueError(cfg.feature_transform)


# ───────────────────────────── engine ─────────────────────────────

class FLRun:
    def __init__(self, cfg: FLConfig, X, y, samples, client_of, device="cuda",
                 selector=None, verbose=False):
        self.cfg, self.device, self.verbose = cfg, device, verbose
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.samples = samples
        self.y = y
        split = np.array([s["split"] for s in samples])
        self.train_mask = torch.tensor(split == "train", device=device)
        self.X = transform_features(X, self.train_mask, cfg, device)
        self.val_idx = torch.tensor(np.where(split == "val")[0], device=device)
        self.test_idx = torch.tensor(np.where(split == "test")[0], device=device)
        self.client_of = list(client_of)
        self.clients = build_clients(samples, client_of, device)
        self.batch_norm = cfg.algorithm == "FedBN"                 # BN kept local, per-client inference
        self.has_bn = cfg.algorithm in ("FedBN", "FedAvgBN")       # FedAvgBN: BN layers present but aggregated
        self.model = make_model(cfg, self.X.shape[1], self.has_bn).to(device)
        self.bn_keys = self.model.bn_keys() if self.batch_norm else set()
        self.global_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        # class weights from pooled training labels (as in the legacy code)
        if cfg.class_weighting:
            ytr = y[self.train_mask].cpu().numpy()
            cnt = np.bincount(ytr, minlength=NUM_CLASSES)
            w = [len(ytr) / (NUM_CLASSES * c) if c > 0 else 1.0 for c in cnt]
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.c_server = None
        self.selector = selector
        self.history = []

    # ── sampling ──
    def select(self, rnd):
        k = self.cfg.clients_per_round
        cl = self.clients
        if self.selector is not None:
            return self.selector.select(cl, k if k is not None else len(cl))
        if k is None or k >= len(cl):
            return cl
        if self.cfg.sampling == "uniform":
            idx = self.rng.choice(len(cl), k, replace=False)
            return [cl[i] for i in sorted(idx)]
        groups = defaultdict(list)
        for i, c in enumerate(cl):
            groups[c.label].append(i)
        chosen = set(int(self.rng.choice(v)) for _, v in sorted(groups.items()))
        rest = [i for i in range(len(cl)) if i not in chosen]
        extra = k - len(chosen)
        if extra > 0 and rest:
            chosen.update(int(i) for i in self.rng.choice(rest, min(extra, len(rest)), replace=False))
        return [cl[i] for i in sorted(chosen)]

    # ── local update ──
    def local_train(self, client):
        cfg, m = self.cfg, self.model
        start = dict(self.global_state)
        if self.batch_norm and client.local_bn is not None:
            start.update(client.local_bn)
        m.load_state_dict(start)
        m.train()
        params = dict(m.named_parameters())
        anchor = {k: v.detach().clone() for k, v in params.items()}
        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
        else:
            opt = torch.optim.SGD(m.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        scaffold = cfg.algorithm == "SCAFFOLD"
        if scaffold:                                   # control variates kept on `device` (perf only)
            cs = self.c_server
            ci = client.c_i if client.c_i is not None else {k: torch.zeros_like(v) for k, v in cs.items()}
        n, bs = client.n_train, cfg.batch_size
        steps = 0
        for _ in range(cfg.local_epochs):
            perm = client.train_idx[torch.randperm(n, device=self.device)]
            if cfg.legacy_drop_last:
                nb = n // bs if n > 1 else 1                       # legacy DataLoader(drop_last=True)
                bounds = [(b * bs, (b + 1) * bs) for b in range(nb)]
            else:                                                  # [v2-2]
                bounds = [(s, min(s + bs, n)) for s in range(0, n, bs)]
                if self.has_bn and len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
                    bounds = bounds[:-1]                           # BN cannot use a size-1 batch
            for a, b in bounds:
                idx = perm[a:b]
                opt.zero_grad(set_to_none=True)
                loss = self.criterion(m(self.X[idx]), self.y[idx])
                if cfg.algorithm == "FedProx":
                    prox = sum(((p - anchor[k]) ** 2).sum() for k, p in params.items())
                    loss = loss + (cfg.mu / 2.0) * prox
                loss.backward()
                if scaffold:
                    for k, p in params.items():
                        if p.grad is not None:
                            p.grad.add_(cs[k] - ci[k])
                opt.step()
                steps += 1
        new_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if self.batch_norm:
            client.local_bn = {k: v.clone() for k, v in new_state.items() if k in self.bn_keys}
        delta_c = None
        if scaffold and steps > 0:
            new_ci, delta_c = {}, {}
            for k, p in params.items():
                # Option II: c_i+ = c_i - c + (x - y) / (K * lr)
                nci = ci[k] - cs[k] + (anchor[k] - p.detach()) / (steps * cfg.lr)
                delta_c[k] = nci - ci[k]
                new_ci[k] = nci
            client.c_i = new_ci
        return new_state, delta_c, steps

    # ── aggregation ──
    def aggregate(self, states, clients):
        keys = [k for k in states[0] if k not in self.bn_keys]
        sizes = [c.n_train for c in clients]
        if self.cfg.aggregation == "sample_weighted":
            weights = [s / sum(sizes) for s in sizes]
        else:                                                   # legacy class-balanced FedAvg
            by = defaultdict(list)
            for i, c in enumerate(clients):
                by[c.label].append(i)
            weights = [0.0] * len(clients)
            for lab, idxs in by.items():
                tot = sum(sizes[i] for i in idxs)
                for i in idxs:
                    weights[i] = (sizes[i] / tot) / len(by)
        new = {}
        for k in keys:
            acc = torch.zeros_like(states[0][k], dtype=torch.float32)
            for st, w in zip(states, weights):
                acc.add_(st[k].float(), alpha=w)
            new[k] = acc.to(states[0][k].dtype)
        for k in self.bn_keys:                                  # placeholder, never used for eval
            new[k] = states[0][k].clone()
        return new

    # ── evaluation ──
    @torch.no_grad()
    def _logits(self, idx, state):
        self.model.load_state_dict(state)
        self.model.eval()
        out = []
        for s in range(0, idx.numel(), 1024):
            out.append(self.model(self.X[idx[s:s + 1024]]))
        return torch.cat(out) if out else torch.empty(0, NUM_CLASSES, device=self.device)

    @torch.no_grad()
    def _fedbn_logits(self, split, bn_snapshot=None):
        """Per-client prediction with each client's own BN state."""
        idx_all, logit_all = [], []
        for c in self.clients:
            idx = c.val_idx if split == "val" else c.test_idx
            if idx.numel() == 0:
                continue
            st = dict(self.global_state)
            bn = (bn_snapshot or {}).get(c.cid, c.local_bn)
            if bn is not None:
                st.update(bn)
            idx_all.append(idx)
            logit_all.append(self._logits(idx, st))
        return torch.cat(idx_all), torch.cat(logit_all)

    def monitor(self):
        if self.batch_norm:
            idx, lg = self._fedbn_logits("val")
            pred = lg.argmax(1)
            corr = (pred == self.y[idx]).float()
            per = []
            cl_of = {}
            for c in self.clients:
                for i in c.val_idx.tolist():
                    cl_of[i] = c.cid
            by = defaultdict(list)
            for i, v in zip(idx.tolist(), corr.tolist()):
                by[cl_of[i]].append(v)
            return float(np.mean([np.mean(v) for v in by.values()]))
        lg = self._logits(self.val_idx, self.global_state)
        return float((lg.argmax(1) == self.y[self.val_idx]).float().mean())

    # [v2-6] The legacy implementation rewarded the selected clients with the change in
    # the mean validation accuracy of the first five selected clients (a client-subset proxy).
    # Protocol v2 uses as reward the increase of the pooled
    # validation accuracy between consecutive rounds, max(0, val_t - val_{t-1}), shared
    # equally by the participating clients.

    # ── main loop ──
    def run(self):
        cfg = self.cfg
        t0 = time.time()
        if cfg.algorithm == "SCAFFOLD":
            self.c_server = {k: torch.zeros_like(v.detach()) for k, v in self.model.named_parameters()}
        best, best_round, best_state, best_bn, no_imp = -1.0, 0, None, None, 0
        max_r = cfg.fixed_rounds or cfg.max_rounds
        prev_val = None
        total_steps = 0
        coverage, sel_counts = [], defaultdict(int)
        for r in range(1, max_r + 1):
            sel = self.select(r)
            coverage.append(len(set(c.label for c in sel)))
            for c in sel:
                sel_counts[c.cid] += 1
            states, dcs = [], []
            for c in sel:
                st, dc, steps = self.local_train(c)
                states.append(st)
                total_steps += steps
                if dc is not None:
                    dcs.append(dc)
            self.global_state = self.aggregate(states, sel)
            if cfg.algorithm == "SCAFFOLD" and dcs:
                N = len(self.clients)
                for k in self.c_server:
                    self.c_server[k] += sum(d[k] for d in dcs) / N
            val = self.monitor()                                    # [v2-1] validation only
            if self.selector is not None:
                if prev_val is not None:
                    self.selector.update(sel, max(0.0, val - prev_val))   # [v2-6] reward = pooled validation delta
                prev_val = val
            self.history.append(val)
            if val > best + cfg.min_delta:
                best, best_round, no_imp = val, r, 0
                best_state = {k: v.clone() for k, v in self.global_state.items()}
                if self.batch_norm:
                    best_bn = {c.cid: ({k: v.clone() for k, v in c.local_bn.items()} if c.local_bn else None)
                               for c in self.clients}
            else:
                no_imp += 1
            if self.verbose and r % 25 == 0:
                print(f"  round {r} val={val:.4f} best={best:.4f}@{best_round}", flush=True)
            if cfg.fixed_rounds is None and no_imp >= cfg.patience and r >= cfg.min_rounds:
                break
        self.global_state = best_state
        if self.batch_norm:
            idx, lg = self._fedbn_logits("test", best_bn)
        else:
            idx, lg = self.test_idx, self._logits(self.test_idx, best_state)
        probs = torch.softmax(lg.float(), 1).cpu().numpy()
        idx = idx.cpu().numpy()
        order = np.argsort(idx)
        idx, probs = idx[order], probs[order]
        y_true = self.y.cpu().numpy()[idx]
        client_ids = np.array([self.samples[i]["client_id"] for i in idx])
        met = classification_metrics(y_true, probs.argmax(1), client_ids)
        # macro accuracy over the TRAINING clients (institutions for the institution partitions)
        pred = probs.argmax(1)
        by = defaultdict(lambda: [0, 0])
        for i, t, p_ in zip(idx, y_true, pred):
            by[self.client_of[i]][0] += int(t == p_); by[self.client_of[i]][1] += 1
        met["per_train_client"] = {c: v for c, v in by.items()}
        met["per_train_client_macro_accuracy"] = float(np.mean([v[0] / v[1] for v in by.values()]))
        hp = n_params(self.model)
        return dict(
            config=asdict(cfg), rounds_run=len(self.history), best_round=best_round,
            best_val=best, val_history=self.history, test=met, n_params=hp,
            bytes_per_model=hp * 4, total_local_steps=total_steps,
            coverage_history=coverage, selection_counts=dict(sel_counts),
            seconds=time.time() - t0,
        ), dict(test_idx=idx, probs=probs.astype(np.float16)), best_state
