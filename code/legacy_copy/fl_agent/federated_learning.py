"""
Federated Learning Algorithms for WSI Analysis
Supports: FedAvg, FedProx, SCAFFOLD (complete), FedBN (true local-BN)

FedPath-Drift components:
  - True FedBN  : BN layers kept strictly local; server aggregates only non-BN params.
  - Full SCAFFOLD: per-client control variates c_i + server variate c with Option-II update.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ─────────────────────────────────────────────
# Feature loading helpers
# ─────────────────────────────────────────────

def load_features_batch(sample_list: List[dict], device: str = "cpu") -> tuple:
    """Load .npy features for a list of samples. Returns (features, labels)."""
    feats, labels = [], []
    for s in sample_list:
        try:
            feat = np.load(s["path"])
            if feat.ndim > 1:
                feat = feat.mean(axis=0)  # aggregate tile features → slide-level
            feats.append(feat)
            labels.append(s["cancer_label"])
        except Exception:
            pass  # skip corrupted files
    if not feats:
        return None, None
    X = torch.tensor(np.array(feats), dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    return X, y


def load_survival_batch(sample_list: List[dict], device: str = "cpu") -> tuple:
    """Load .npy features + survival labels. Returns (X, os_time, os_status)."""
    feats, times, events = [], [], []
    for s in sample_list:
        try:
            feat = np.load(s["path"])
            if feat.ndim > 1:
                feat = feat.mean(axis=0)
            feats.append(feat)
            times.append(s["os_time"])
            events.append(s["os_status"])
        except Exception:
            pass
    if not feats:
        return None, None, None
    X = torch.tensor(np.array(feats), dtype=torch.float32).to(device)
    t = torch.tensor(times, dtype=torch.float32).to(device)
    e = torch.tensor(events, dtype=torch.float32).to(device)
    return X, t, e


def make_dataloader(samples: List[dict], batch_size: int = 32, shuffle: bool = True, device: str = "cpu"):
    X, y = load_features_batch(samples, device=device)
    if X is None:
        return None
    dataset = TensorDataset(X, y)
    # drop_last prevents BatchNorm errors on single-sample last batches during training
    drop_last = shuffle and len(dataset) > 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def make_survival_dataloader(samples: List[dict], batch_size: int = 32,
                              shuffle: bool = True, device: str = "cpu"):
    """DataLoader for survival prediction (returns X, os_time, os_status)."""
    X, t, e = load_survival_batch(samples, device=device)
    if X is None:
        return None
    dataset = TensorDataset(X, t, e)
    drop_last = shuffle and len(dataset) > 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# ─────────────────────────────────────────────
# FedAvg
# ─────────────────────────────────────────────

def fed_avg(client_weights: List[dict], client_sizes: List[int]) -> dict:
    """Weighted average of client model weights."""
    total = sum(client_sizes)
    avg = {}
    for key in client_weights[0]:
        avg[key] = sum(
            w[key] * (n / total) for w, n in zip(client_weights, client_sizes)
        )
    return avg


# ─────────────────────────────────────────────
# Client trainer
# ─────────────────────────────────────────────

class FLClient:
    """
    A single federated learning client.
    Holds local data and performs local training.

    FedBN: stores local BN state (all BN params) between rounds.
    SCAFFOLD: stores per-client control variate c_i between rounds.
    """

    def __init__(
        self,
        client_id: str,
        train_samples: List[dict],
        val_samples: List[dict],
        model: nn.Module,
        cfg: dict,
        test_samples: List[dict] = None,
    ):
        self.client_id = client_id
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples or []
        self.model = copy.deepcopy(model)
        self.cfg = cfg
        self.device = cfg["experiment"]["device"]
        self.n_train = len(train_samples)

        # FedBN: per-client BN state (all BN params: weight, bias, running_mean/var, nbatch)
        self.local_bn_params: dict = {}
        # SCAFFOLD: per-client control variate c_i (keyed by named_parameter name)
        self.c_i: dict = {}

    def _bn_keys(self) -> set:
        """Return set of BN param names from the model (empty if no BN layers)."""
        if hasattr(self.model, "get_bn_param_names"):
            return self.model.get_bn_param_names()
        return set()

    # ── Local training ────────────────────────────────────────────

    def local_train(
        self,
        global_weights: dict,
        task: str = "cancer_type_classification",
        class_weights: torch.Tensor = None,
        server_c: dict = None,
    ) -> Tuple[dict, dict]:
        """
        Perform local training starting from global_weights.

        FedBN: merges global non-BN params with client's own BN state.
        SCAFFOLD: applies gradient correction (c - c_i) each step; computes
                  delta_c_i = c_i_new - c_i via Option II after training.

        Returns:
            (updated_weights, aux_dict)
            aux_dict = {"delta_c": {name: tensor}} for SCAFFOLD, {} otherwise.
        """
        algo    = self.cfg["federated"]["algorithm"]
        lr      = self.cfg["federated"]["local_lr"]
        mu      = self.cfg["federated"].get("mu", 0.01)
        bn_keys = self._bn_keys()

        # ── Set initial weights ──────────────────────────────────
        if algo == "FedBN" and bn_keys and self.local_bn_params:
            # Merge: non-BN from global, BN from client's own state
            merged = dict(global_weights)
            merged.update(self.local_bn_params)
            self.model.set_weights(merged)
        else:
            self.model.set_weights(global_weights)

        self.model.train()
        self.model.to(self.device)

        # Snapshot the starting model's named parameters for:
        #   - FedProx proximal term  (global_params, keyed by state_dict key)
        #   - SCAFFOLD Option-II delta_c (global_named, keyed by named_parameter name)
        # The model is already loaded with the correct starting weights above.
        global_params = {k: v.clone().to(self.device)
                         for k, v in global_weights.items()
                         if not k.endswith("num_batches_tracked")}
        global_named  = {name: p.data.clone().to(self.device)
                         for name, p in self.model.named_parameters()}

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        # Pre-fetch SCAFFOLD control variates to device
        if algo == "SCAFFOLD" and server_c is not None:
            c_server = {k: v.clone().to(self.device) for k, v in server_c.items()}
            c_client = {k: v.clone().to(self.device) for k, v in self.c_i.items()} \
                       if self.c_i else {}
        else:
            c_server = c_client = {}

        total_steps = 0
        loss_sum_for_avg = 0.0       # for aux["local_loss"]
        loss_count_for_avg = 0

        # ── Survival task ────────────────────────────────────────
        if task == "survival_prediction":
            from models.classifiers import cox_loss
            loader = make_survival_dataloader(
                self.train_samples,
                batch_size=len(self.train_samples),  # full batch for Cox
                shuffle=False,
                device=self.device,
            )
            if loader is None:
                return global_weights, {}

            for epoch in range(self.cfg["federated"]["local_epochs"]):
                for X, t, e in loader:
                    optimizer.zero_grad()
                    risk = self.model(X)
                    loss = cox_loss(risk, t, e)
                    if algo == "FedProx":
                        prox = sum(
                            torch.norm(p - global_params[k]) ** 2
                            for k, p in self.model.named_parameters()
                            if k in global_params
                        )
                        loss = loss + (mu / 2) * prox
                    if not torch.isnan(loss):
                        loss.backward()
                        if algo == "SCAFFOLD" and c_server:
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    cs = c_server.get(name, torch.zeros_like(param.grad))
                                    ci = c_client.get(name, torch.zeros_like(param.grad))
                                    param.grad.add_(cs - ci)
                        optimizer.step()
                        total_steps += 1
                        loss_sum_for_avg += float(loss.detach().item())
                        loss_count_for_avg += 1

        # ── Classification task ──────────────────────────────────
        else:
            loader = make_dataloader(
                self.train_samples,
                batch_size=self.cfg["experiment"]["batch_size"],
                shuffle=True,
                device=self.device,
            )
            if loader is None:
                return global_weights, {}

            for epoch in range(self.cfg["federated"]["local_epochs"]):
                for X, y in loader:
                    optimizer.zero_grad()
                    logits = self.model(X)
                    loss = criterion(logits, y)
                    if algo == "FedProx":
                        prox = sum(
                            torch.norm(p - global_params[k]) ** 2
                            for k, p in self.model.named_parameters()
                            if k in global_params
                        )
                        loss = loss + (mu / 2) * prox
                    loss.backward()
                    if algo == "SCAFFOLD" and c_server:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                cs = c_server.get(name, torch.zeros_like(param.grad))
                                ci = c_client.get(name, torch.zeros_like(param.grad))
                                param.grad.add_(cs - ci)
                    optimizer.step()
                    total_steps += 1
                    loss_sum_for_avg += float(loss.detach().item())
                    loss_count_for_avg += 1

        all_weights = self.model.get_weights()

        # ── FedBN: save updated BN state ─────────────────────────
        if algo == "FedBN" and bn_keys:
            self.local_bn_params = {
                k: v.clone() for k, v in all_weights.items() if k in bn_keys
            }

        # ── SCAFFOLD: compute delta_c (Option II) ────────────────
        aux = {}
        if algo == "SCAFFOLD" and c_server and total_steps > 0:
            new_c_i = {}
            delta_c = {}
            for name, param in self.model.named_parameters():
                x_k   = global_named[name]       # global param at round start
                y_k   = param.data               # local param after training
                ci_k  = c_client.get(name, torch.zeros_like(param.data))
                cs_k  = c_server.get(name, torch.zeros_like(param.data))
                # c_i+ = c_i - c + (x - y) / (K * lr)
                new_ci = ci_k - cs_k + (x_k - y_k) / (total_steps * lr)
                delta_c[name]  = (new_ci - ci_k).detach().cpu()
                new_c_i[name]  = new_ci.detach().cpu()
            self.c_i = new_c_i
            aux["delta_c"] = delta_c

        # ── Auxiliary stats for adaptive selectors (Oort/LossBased/FedCor) ──
        if loss_count_for_avg > 0:
            aux["local_loss"] = loss_sum_for_avg / loss_count_for_avg

        # Weight-delta L2 norm vs starting global weights (cheap; for FedCor)
        try:
            sq = 0.0
            for k, w_new in all_weights.items():
                if k.endswith("num_batches_tracked"):
                    continue
                w_old = global_weights.get(k)
                if w_old is None:
                    continue
                sq += float(((w_new - w_old) ** 2).sum().item())
            aux["delta_norm"] = sq ** 0.5
        except Exception:
            aux["delta_norm"] = 0.0

        return all_weights, aux

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(self, weights: dict, split: str = "val") -> dict:
        """
        Evaluate model on val / test / train split.

        FedBN: merges global non-BN weights with client's own BN state before eval,
               so each client's BN running stats reflect its local data distribution.
        """
        algo    = self.cfg["federated"]["algorithm"]
        bn_keys = self._bn_keys()

        if split == "val":
            samples = self.val_samples
        elif split == "test":
            samples = self.test_samples
        else:
            samples = self.train_samples

        # FedBN: overlay client's local BN state
        if algo == "FedBN" and bn_keys and self.local_bn_params:
            merged = dict(weights)
            merged.update(self.local_bn_params)
            self.model.set_weights(merged)
        else:
            self.model.set_weights(weights)

        self.model.eval()
        self.model.to(self.device)

        # Survival task: skip expensive C-index per-round eval, return placeholder
        if samples and "os_time" in samples[0]:
            return {"accuracy": 0.5, "loss": 0.0, "n": len(samples)}

        loader = make_dataloader(samples, batch_size=64, shuffle=False, device=self.device)
        if loader is None:
            return {"accuracy": 0.0, "loss": float("inf"), "n": 0}

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in loader:
                logits = self.model(X)
                loss   = criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                preds  = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += X.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        acc = correct / total if total > 0 else 0.0
        return {
            "accuracy": acc,
            "loss": total_loss / total if total > 0 else float("inf"),
            "n": total,
            "preds": all_preds,
            "labels": all_labels,
        }


# ─────────────────────────────────────────────
# FL Server / Orchestrator
# ─────────────────────────────────────────────

class FLServer:
    """
    Central FL server coordinating training across clients.

    Implements FedAvg, FedProx, FedBN (true local-BN), and SCAFFOLD (complete).
    """

    def __init__(self, global_model: nn.Module, cfg: dict,
                 class_weights: torch.Tensor = None,
                 client_selector=None,
                 n_total_clients: int = 0):
        self.global_model    = global_model
        self.cfg             = cfg
        self.global_weights  = global_model.get_weights()
        self.history         = []
        self.class_weights   = class_weights
        self.client_selector = client_selector
        # SCAFFOLD server control variate (initialised on first aggregate call)
        self.c: Optional[dict] = None
        # Total clients across all rounds (for SCAFFOLD c update scale)
        self.n_total_clients = n_total_clients

    def select_clients(self, clients: List[FLClient]) -> List[FLClient]:
        """
        Sample clients for this round.
        stratified_sampling=True (default): ensure ≥1 client per cancer type,
        then fill remaining slots randomly — prevents majority-class collapse.
        """
        k = self.cfg["federated"].get("clients_per_round")
        if k is None or k >= len(clients):
            return clients

        use_stratified = self.cfg["federated"].get("stratified_sampling", True)
        if not use_stratified:
            idx = np.random.choice(len(clients), k, replace=False)
            return [clients[i] for i in idx]

        from collections import defaultdict
        groups = defaultdict(list)
        for i, c in enumerate(clients):
            label = c.train_samples[0]["cancer_label"] if c.train_samples else -1
            groups[label].append(i)

        selected_idx = set()
        for label, idxs in groups.items():
            if label < 0:
                continue
            selected_idx.add(np.random.choice(idxs))

        remaining = [i for i in range(len(clients)) if i not in selected_idx]
        extra_needed = k - len(selected_idx)
        if extra_needed > 0 and remaining:
            extra = np.random.choice(remaining,
                                     min(extra_needed, len(remaining)),
                                     replace=False)
            selected_idx.update(extra)

        return [clients[i] for i in sorted(selected_idx)]

    def aggregate(
        self,
        client_weights: List[dict],
        client_sizes:   List[int],
        client_labels:  List[int]  = None,
        aux_list:       List[dict] = None,
    ) -> dict:
        """
        Aggregate client weights.

        FedBN: averages only non-BN parameters; BN params are never aggregated.
        SCAFFOLD: updates server control variate c from per-client delta_c.
        """
        algo         = self.cfg["federated"]["algorithm"]
        use_balanced = self.cfg["federated"].get("stratified_sampling", False)

        # ── Determine BN keys from the global model ───────────────
        bn_keys: set = set()
        if algo == "FedBN" and hasattr(self.global_model, "get_bn_param_names"):
            bn_keys = self.global_model.get_bn_param_names()

        # ── Weighted averaging ────────────────────────────────────
        if bn_keys:
            # Average only non-BN params
            non_bn_w = [{k: v for k, v in w.items() if k not in bn_keys}
                        for w in client_weights]
            non_bn_n = client_sizes
            if use_balanced and client_labels is not None:
                base_non_bn = self._balanced_avg(non_bn_w, non_bn_n, client_labels)
            else:
                base_non_bn = fed_avg(non_bn_w, non_bn_n)
            # BN keys: keep first client's values as structural placeholder only
            # (each client will override with its own local_bn_params at eval time)
            base = dict(base_non_bn)
            for k in bn_keys:
                if k in client_weights[0]:
                    base[k] = client_weights[0][k].clone()
        else:
            if use_balanced and client_labels is not None:
                base = self._balanced_avg(client_weights, client_sizes, client_labels)
            else:
                base = fed_avg(client_weights, client_sizes)

        # ── SCAFFOLD: update server control variate c ─────────────
        if algo == "SCAFFOLD" and aux_list:
            delta_cs = [aux["delta_c"] for aux in aux_list if "delta_c" in aux]
            if delta_cs:
                n_total = max(self.n_total_clients, len(delta_cs))
                if self.c is None:
                    self.c = {k: torch.zeros_like(v)
                               for k, v in delta_cs[0].items()}
                for key in self.c:
                    # c ← c + (1/N) * Σ delta_c_i  (SCAFFOLD paper eq.)
                    # delta_c tensors are always CPU (.detach().cpu() in local_train);
                    # self.c is also CPU — no device mismatch.
                    total_delta = sum(
                        dc.get(key, torch.zeros_like(self.c[key])).to(self.c[key].device)
                        for dc in delta_cs
                    )
                    self.c[key] = self.c[key] + total_delta / n_total

        return base

    def _balanced_avg(self, client_weights, client_sizes, client_labels):
        """
        Class-balanced FedAvg: each cancer type contributes equally.
        Within a class, weight by n_train. Across classes, equal weight.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for w, n, lbl in zip(client_weights, client_sizes, client_labels):
            groups[lbl].append((w, n))

        n_classes = len(groups)
        avg = {k: client_weights[0][k].clone().zero_() for k in client_weights[0]}

        for lbl, group in groups.items():
            total_n    = sum(n for _, n in group)
            class_frac = 1.0 / n_classes
            for w, n in group:
                local_frac = (n / total_n) * class_frac
                for k in avg:
                    avg[k] = avg[k] + w[k] * local_frac

        return avg

    def run_round(self, clients: List[FLClient], round_num: int, task: str) -> dict:
        """Execute one FL round: select → local train → aggregate."""
        algo = self.cfg["federated"]["algorithm"]

        # ── Stress-test gate (E4): late-arrival masks the client pool ──
        # If late_arrival is configured, the last N clients are hidden from
        # the selector until round R. All other stress effects are applied
        # AFTER selection (below).
        stress = self.cfg["federated"].get("stress_test", {}) or {}
        if stress.get("enabled"):
            late = stress.get("late_arrival") or {}
            late_round = late.get("round", 0)
            late_n     = late.get("n_late", 0)
            if late_n > 0 and round_num < late_round:
                # Hide last late_n clients (deterministic order from caller)
                visible_n = max(1, len(clients) - late_n)
                clients = clients[:visible_n]

        # Adaptive (UCB / PathologyAware) or default (stratified/random) selection
        if self.client_selector is not None:
            k = self.cfg["federated"].get("clients_per_round")
            k = k if k is not None else len(clients)
            selected = self.client_selector.select(clients, k)
        else:
            selected = self.select_clients(clients)

        # ── Stress-test gate (E4): per-round dropout / sticky straggler ──
        if stress.get("enabled"):
            try:
                rng = np.random.default_rng(round_num + 991)  # deterministic
                # Random per-round dropout
                p_drop = float(stress.get("dropout_prob", 0.0))
                if p_drop > 0:
                    keep = rng.random(len(selected)) > p_drop
                    selected = [c for c, k_ in zip(selected, keep) if k_]
                # Sticky straggler: a fixed fraction always slow → drop them
                # this round with probability 1 - their availability rate.
                p_strag_frac = float(stress.get("straggler_frac", 0.0))
                p_strag_avail = float(stress.get("straggler_avail", 0.1))
                if p_strag_frac > 0 and selected:
                    # Hash-based sticky tag (same client = same straggler status across rounds)
                    is_strag = [(hash(c.client_id) % 1000) / 1000.0 < p_strag_frac
                                for c in selected]
                    keep = [(not s) or (rng.random() < p_strag_avail)
                            for s in is_strag]
                    selected = [c for c, k_ in zip(selected, keep) if k_]
                # Guarantee at least one client (avoid empty round)
                if not selected:
                    selected = [clients[0]]
            except Exception as ex:
                # Stress-test code must never crash the main loop
                print(f"  [stress] WARN: {ex} — falling back to no stress this round")

        print(f"  [Round {round_num}] Training {len(selected)} clients...", end=" ", flush=True)

        # SCAFFOLD: initialise server control variate c to zeros on first round.
        # Kept on CPU so it is device-agnostic; local_train moves it to the
        # client device via .to(self.device) before use.
        if algo == "SCAFFOLD" and self.c is None:
            self.c = {name: torch.zeros(p.shape)   # CPU, float32
                      for name, p in self.global_model.named_parameters()}

        t0 = time.time()
        updated_weights, sizes, labels, aux_list = [], [], [], []
        server_c = self.c if algo == "SCAFFOLD" else None

        for client in selected:
            w, aux = client.local_train(
                self.global_weights,
                task=task,
                class_weights=self.class_weights,
                server_c=server_c,
            )
            updated_weights.append(w)
            sizes.append(client.n_train)
            lbl = client.train_samples[0]["cancer_label"] if client.train_samples else -1
            labels.append(lbl)
            aux_list.append(aux)

        self.global_weights = self.aggregate(updated_weights, sizes, labels, aux_list)
        self.global_model.set_weights(self.global_weights)

        # Monitor on a few clients
        eval_clients = selected[:min(5, len(selected))]
        accs = [c.evaluate(self.global_weights, "val")["accuracy"] for c in eval_clients]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        elapsed = time.time() - t0

        # Update bandit rewards based on accuracy delta + per-client aux
        if self.client_selector is not None and self.history:
            prev_acc = self.history[-1].get("val_accuracy", 0.0)
            delta = mean_acc - prev_acc
            client_losses = {c.client_id: aux.get("local_loss", 0.0)
                             for c, aux in zip(selected, aux_list)
                             if aux.get("local_loss") is not None}
            client_delta_norms = {c.client_id: aux.get("delta_norm", 0.0)
                                  for c, aux in zip(selected, aux_list)
                                  if aux.get("delta_norm") is not None}
            self.client_selector.update(
                selected, delta,
                client_losses=client_losses,
                client_delta_norms=client_delta_norms,
            )

        round_log = {
            "round": round_num,
            "num_clients": len(selected),
            "val_accuracy": mean_acc,
            "elapsed_s": elapsed,
        }
        self.history.append(round_log)
        print(f"val_acc={mean_acc:.4f}  ({elapsed:.1f}s)")
        return round_log

    def train(self, clients: List[FLClient], task: str = "cancer_type_classification",
              global_test: list = None) -> List[dict]:
        """
        Run FL training with optional early stopping.

        Early stopping config (under cfg["federated"]):
          early_stopping_patience : int   — rounds with no improvement (default 20)
          early_stopping_min_delta: float — minimum improvement threshold (default 0.001)
          restore_best_weights    : bool  — restore best weights on stop (default True)
        """
        num_rounds = self.cfg["federated"]["num_rounds"]
        device     = self.cfg["experiment"]["device"]

        es_cfg       = self.cfg["federated"]
        patience     = es_cfg.get("early_stopping_patience", 50)
        min_delta    = es_cfg.get("early_stopping_min_delta", 0.001)
        min_rounds   = es_cfg.get("early_stopping_min_rounds", 100)
        restore_best = es_cfg.get("restore_best_weights", True)

        best_acc         = -1.0
        best_weights     = None
        best_round       = 0
        no_improve_count = 0

        # Global validation loader for per-round monitoring.
        # The caller (AutoFLAgent.train) passes the pooled per-client *validation*
        # splits here, NOT the held-out test set, so checkpoint selection is based
        # on data that is strictly separate from the final test evaluation.
        global_val_loader = None
        if global_test:
            global_val_loader = make_dataloader(global_test, 64, False, device)

        print(f"\n[FL Server] Starting {self.cfg['federated']['algorithm']} "
              f"— max {num_rounds} rounds  "
              f"(early stop: patience={patience}, min_delta={min_delta})")

        for r in range(1, num_rounds + 1):
            log = self.run_round(clients, r, task)

            # Monitor accuracy for early stopping.
            # FedBN: the server's global model holds placeholder BN stats from
            # client[0], which are meaningless on the pooled multi-cancer val set.
            # Use per-client val_accuracy (already computed in run_round) instead.
            algo_now = self.cfg["federated"]["algorithm"]
            if global_val_loader is not None and algo_now != "FedBN":
                self.global_model.eval()
                self.global_model.to(device)
                correct, total = 0, 0
                with torch.no_grad():
                    for X, y in global_val_loader:
                        correct += (self.global_model(X).argmax(1) == y).sum().item()
                        total   += X.size(0)
                monitor_acc = correct / total if total > 0 else 0.0
                self.history[-1]["global_val_acc"] = monitor_acc
                if r % 10 == 0:
                    print(f"    >>> Round {r} global val acc: {monitor_acc:.4f}")
            else:
                monitor_acc = log["val_accuracy"]

            if monitor_acc > best_acc + min_delta:
                best_acc         = monitor_acc
                best_round       = r
                best_weights     = copy.deepcopy(self.global_weights)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience and r >= min_rounds:
                print(f"\n[Early Stop] No improvement for {patience} rounds "
                      f"(round {r} ≥ min_rounds {min_rounds}). "
                      f"Best val_acc={best_acc:.4f} at round {best_round}. Stopping.")
                break

        if restore_best and best_weights is not None:
            self.global_weights = best_weights
            self.global_model.set_weights(best_weights)
            print(f"[Early Stop] Restored best weights from round {best_round} "
                  f"(val_acc={best_acc:.4f})")

        print(f"[FL Server] Training complete. Total rounds: {len(self.history)}")
        return self.history

    # ── Evaluation ────────────────────────────────────────────────

    def global_evaluate(self, test_samples: List[dict], device: str = "cuda") -> dict:
        """Evaluate global model on the held-out global test set (non-FedBN)."""
        self.global_model.eval()
        self.global_model.to(device)
        loader = make_dataloader(test_samples, batch_size=64, shuffle=False, device=device)
        if loader is None:
            return {}

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in loader:
                logits = self.global_model(X)
                loss   = criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                preds  = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += X.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        result = {
            "test_accuracy": correct / total if total > 0 else 0.0,
            "test_loss": total_loss / total if total > 0 else float("inf"),
            "n_test": total,
            "predictions": all_preds,
            "labels": all_labels,
        }
        print(f"[Global Eval] Test accuracy: {result['test_accuracy']:.4f}  (n={total})")
        return result

    def global_evaluate_fedbn(self, clients: List["FLClient"],
                               device: str = "cuda") -> dict:
        """
        FedBN evaluation: each client evaluates its own test split using its
        local BN state.  Reports per-client accuracy, macro-average, and
        worst-10th-percentile accuracy (equity metric).
        """
        per_client: dict = {}
        total_correct, total_n = 0, 0

        for client in clients:
            if not client.test_samples:
                continue
            res = client.evaluate(self.global_weights, split="test")
            if res["n"] == 0:
                continue
            per_client[client.client_id] = res
            total_correct += int(res["accuracy"] * res["n"])
            total_n       += res["n"]

        accs = [v["accuracy"] for v in per_client.values()]
        macro_avg     = float(np.mean(accs)) if accs else 0.0
        worst_10pct   = float(np.percentile(accs, 10)) if accs else 0.0
        micro_avg     = total_correct / total_n if total_n > 0 else 0.0

        print(f"[FedBN Eval] macro_avg={macro_avg:.4f}  "
              f"micro_avg={micro_avg:.4f}  "
              f"worst_10pct={worst_10pct:.4f}  (n_clients={len(accs)})")
        return {
            "test_accuracy": macro_avg,        # report macro as primary metric
            "macro_accuracy": macro_avg,
            "micro_accuracy": micro_avg,
            "worst_10pct_accuracy": worst_10pct,
            "per_client": per_client,
            "n_test": total_n,
        }

    def global_evaluate_survival(self, test_samples: List[dict], device: str = "cuda") -> dict:
        """Evaluate global survival model with C-index."""
        from models.classifiers import concordance_index
        self.global_model.eval()
        self.global_model.to(device)
        loader = make_survival_dataloader(test_samples, batch_size=64, shuffle=False, device=device)
        if loader is None:
            return {}
        all_risk, all_time, all_event = [], [], []
        with torch.no_grad():
            for X, t, e in loader:
                risk = self.global_model(X)
                all_risk.extend(risk.cpu().numpy())
                all_time.extend(t.cpu().numpy())
                all_event.extend(e.cpu().numpy())
        cindex = concordance_index(all_risk, all_time, all_event)
        print(f"[Global Eval] C-index: {cindex:.4f}  "
              f"(n={len(all_risk)}, events={int(sum(all_event))})")
        return {"c_index": cindex, "n_test": len(all_risk), "n_events": int(sum(all_event))}
