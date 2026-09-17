"""
Bandit-based adaptive client selection for federated learning.

UCBClientSelector implements UCB1 (Upper Confidence Bound) to adaptively
select which clients participate each FL round based on their historical
contribution to global model improvement.

Reference:
  Auer et al. (2002). Finite-time analysis of the multiarmed bandit problem.
  Machine Learning 47:235–256.

Usage:
    selector = UCBClientSelector(client_ids=[c.client_id for c in clients],
                                 c=2.0)
    # Each round:
    selected = selector.select(clients, k=20)
    ... train and evaluate ...
    reward = new_acc - prev_acc          # accuracy delta
    selector.update(selected, reward)
"""

import numpy as np
from typing import List


class UCBClientSelector:
    """
    UCB1-based adaptive client selector.

    Each client is treated as a bandit arm.  The UCB score balances:
      - exploitation: historically high-reward clients
      - exploration:  rarely selected clients

    UCB score for client i at round t:
        score_i = mu_i + c * sqrt(log(t) / n_i)

    where mu_i is the running mean reward, n_i is the selection count,
    and c is the exploration coefficient (default 2.0).

    Reward definition:
        After each round, reward = max(0, global_val_acc_after
                                        - global_val_acc_before)
        This reward is split equally among the k selected clients.
        Negative deltas are clipped to 0 (pure reward, no penalty).
    """

    def __init__(self, client_ids: List[str], c: float = 2.0):
        """
        Args:
            client_ids: list of unique string identifiers for all clients
            c: exploration coefficient; higher c → more exploration
        """
        self.c = c
        self.t = 0                                       # global round counter
        self.counts  = {cid: 0   for cid in client_ids} # n_i
        self.values  = {cid: 0.0 for cid in client_ids} # mu_i (mean reward)

    def register(self, client_ids: List[str]):
        """Register new client IDs (no-op for already known IDs)."""
        for cid in client_ids:
            if cid not in self.counts:
                self.counts[cid]  = 0
                self.values[cid]  = 0.0

    def select(self, clients, k: int):
        """
        Select k clients by UCB1 score.

        Unvisited clients (n_i == 0) receive infinite score to ensure
        every client is explored at least once before exploitation begins.

        Args:
            clients: list of FLClient objects with attribute .client_id
            k:       number of clients to select

        Returns:
            List of k FLClient objects sorted by descending UCB score.
        """
        self.t += 1
        if k >= len(clients):
            return clients

        cids = [c.client_id for c in clients]
        self.register(cids)

        scores = {}
        for c in clients:
            cid = c.client_id
            n   = self.counts[cid]
            if n == 0:
                scores[cid] = float("inf")          # force initial exploration
            else:
                scores[cid] = (self.values[cid]
                               + self.c * np.sqrt(np.log(self.t) / n))

        sorted_clients = sorted(clients,
                                key=lambda c: -scores[c.client_id])
        return sorted_clients[:k]

    def update(self, selected_clients, reward: float, **kwargs):
        """
        Update running-mean estimates for the selected clients.

        Uses incremental mean update to avoid storing all history:
            mu_i ← mu_i + (r - mu_i) / n_i

        Args:
            selected_clients: list of FLClient objects that participated
            reward:           scalar accuracy delta for this round
            **kwargs:         forward-compatible (ignored here; used by Oort/LossBased/FedCor)
        """
        n_sel = max(1, len(selected_clients))
        per_client_reward = max(0.0, reward) / n_sel   # clip negatives

        for client in selected_clients:
            cid = client.client_id
            self.counts[cid] += 1
            n = self.counts[cid]
            # Incremental mean: O(1) memory
            self.values[cid] += (per_client_reward - self.values[cid]) / n

    def top_clients(self, n: int = 10):
        """Return the n highest-value clients (for analysis / logging)."""
        ranked = sorted(self.values.items(), key=lambda kv: -kv[1])
        return ranked[:n]

    def summary_stats(self) -> dict:
        """Return summary statistics for logging."""
        vals = list(self.values.values())
        cnts = list(self.counts.values())
        return {
            "t": self.t,
            "n_explored": sum(1 for n in cnts if n > 0),
            "n_total":    len(cnts),
            "mean_reward": float(np.mean(vals)) if vals else 0.0,
            "max_reward":  float(np.max(vals))  if vals else 0.0,
            "min_count":   int(min(cnts))        if cnts else 0,
            "max_count":   int(max(cnts))        if cnts else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PathologyAwareSelector  (FedPath-Drift)
# ─────────────────────────────────────────────────────────────────────────────

class PathologyAwareSelector(UCBClientSelector):
    """
    Multi-objective constrained client selector for pathology FL (FedPath-Drift).

    Score:
        score_i(t) = utility_i
                   + beta  * sqrt(log(t) / n_i)          [exploration]
                   + lam   * feature_diversity_i          [feature shift]
                   - rho   * instability_i                [reward variance penalty]

    Feature diversity: 1 - cosine_similarity(client_feature_mean, global_prototype).
    Instability: variance of the client's recent per-round rewards (rolling window).

    Selection strategy (constrained greedy):
      Phase 1 — guarantee at least one client per cancer type (by highest score).
      Phase 2 — fill remaining slots from unselected clients by score.

    Reference:
      FedCor (Tang et al., NeurIPS 2022); Oort (Lai et al., OSDI 2021).
    """

    def __init__(
        self,
        client_ids: List[str],
        c:    float = 2.0,
        beta: float = 1.0,
        lam:  float = 0.5,
        rho:  float = 0.2,
        window: int = 10,
    ):
        """
        Args:
            c     : UCB exploration coefficient (passed to parent)
            beta  : exploration weight in multi-objective score
            lam   : feature-diversity weight
            rho   : instability penalty weight
            window: number of recent rounds used for instability estimate
        """
        super().__init__(client_ids, c=c)
        self.beta   = beta
        self.lam    = lam
        self.rho    = rho
        self.window = window

        # {cid: np.ndarray} — slide-level feature mean for each client
        self.feature_means: dict   = {}
        # Global centroid across all clients (updated after register_feature_means)
        self.global_prototype: np.ndarray = None
        # {cid: list} — recent per-client reward history (capped at window)
        self.reward_history: dict  = {}
        # {cid: int} — dominant cancer label per client
        self.cancer_map: dict      = {}

    # ── One-time setup ────────────────────────────────────────────

    def register_feature_means(self, clients, max_slides: int = 30):
        """
        Compute and cache the mean feature vector for each client.
        Only the first `max_slides` slides per client are loaded to bound I/O cost.

        Also maps each client to its dominant cancer label for constrained selection.
        """
        all_means = []
        for client in clients:
            feats = []
            for s in client.train_samples[:max_slides]:
                try:
                    feat = np.load(s["path"])
                    if feat.ndim > 1:
                        feat = feat.mean(axis=0)
                    feats.append(feat.astype(np.float32))
                except Exception:
                    pass
            if feats:
                mean = np.mean(feats, axis=0)
                self.feature_means[client.client_id] = mean
                all_means.append(mean)
            # Dominant cancer label
            lbl = (client.train_samples[0].get("cancer_label", -1)
                   if client.train_samples else -1)
            self.cancer_map[client.client_id] = lbl

        if all_means:
            self.global_prototype = np.mean(all_means, axis=0)
        print(f"[PathologyAwareSelector] Feature means registered for "
              f"{len(self.feature_means)} clients.")

    # ── Score components ──────────────────────────────────────────

    def _feature_diversity(self, cid: str) -> float:
        """Cosine distance between client feature mean and global prototype."""
        if cid not in self.feature_means or self.global_prototype is None:
            return 0.0
        cm  = self.feature_means[cid]
        gp  = self.global_prototype
        dot = np.dot(cm, gp)
        nrm = np.linalg.norm(cm) * np.linalg.norm(gp)
        return float(1.0 - dot / (nrm + 1e-8))

    def _instability(self, cid: str) -> float:
        """Variance of the client's recent rewards (0 if fewer than 2 observations)."""
        hist = self.reward_history.get(cid, [])
        return float(np.var(hist)) if len(hist) >= 2 else 0.0

    def _score(self, cid: str) -> float:
        n = self.counts[cid]
        if n == 0:
            return float("inf")
        utility     = self.values[cid]
        exploration = self.beta * np.sqrt(np.log(self.t) / n)
        feat_div    = self.lam  * self._feature_diversity(cid)
        instab      = self.rho  * self._instability(cid)
        return utility + exploration + feat_div - instab

    # ── Selection ─────────────────────────────────────────────────

    def select(self, clients, k: int):
        self.t += 1
        if k >= len(clients):
            return clients

        cids = [c.client_id for c in clients]
        self.register(cids)

        scores = {c.client_id: self._score(c.client_id) for c in clients}

        # Phase 1: cancer-coverage constraint — one client per type
        from collections import defaultdict
        cancer_groups = defaultdict(list)
        for c in clients:
            cancer_groups[self.cancer_map.get(c.client_id, -1)].append(c)

        selected     = []
        selected_ids = set()
        for cancer, group in sorted(cancer_groups.items()):
            if cancer < 0:
                continue
            best = max(group, key=lambda c: scores.get(c.client_id, -float("inf")))
            if best.client_id not in selected_ids:
                selected.append(best)
                selected_ids.add(best.client_id)
            if len(selected) >= k:
                break

        # Phase 2: fill remaining slots by multi-objective score
        remaining = sorted(
            [c for c in clients if c.client_id not in selected_ids],
            key=lambda c: -scores.get(c.client_id, 0.0),
        )
        for c in remaining:
            if len(selected) >= k:
                break
            selected.append(c)
            selected_ids.add(c.client_id)

        return selected[:k]

    # ── Reward update ─────────────────────────────────────────────

    def update(self, selected_clients, reward: float, **kwargs):
        """Update running-mean utility and rolling reward history."""
        super().update(selected_clients, reward, **kwargs)
        per_client_reward = max(0.0, reward) / max(1, len(selected_clients))
        for client in selected_clients:
            cid = client.client_id
            hist = self.reward_history.setdefault(cid, [])
            hist.append(per_client_reward)
            if len(hist) > self.window:
                hist.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# OortSelector  (Lai et al., OSDI 2021)
# ─────────────────────────────────────────────────────────────────────────────

class OortSelector(UCBClientSelector):
    """
    Oort-style adaptive client selector (Lai et al. OSDI 2021, simplified).

    Statistical utility:
        utility_i = sqrt(N_i) * loss_i

    Final score:
        score_i(t) = alpha * utility_i + c * sqrt(log(t) / n_i)

    where N_i is the number of training samples on client i, loss_i is the
    most recent local training loss, and the second term is the standard
    UCB exploration bonus. Until a client has been visited at least once
    (or before any loss has been reported), the selector falls back to
    UCB1 behaviour.
    """

    def __init__(self, client_ids, c: float = 2.0, alpha: float = 1.0):
        super().__init__(client_ids, c=c)
        self.alpha = alpha
        self.client_losses = {cid: 0.0 for cid in client_ids}
        self.client_sizes  = {cid: 1   for cid in client_ids}
        self._has_loss     = {cid: False for cid in client_ids}

    def register(self, client_ids):
        for cid in client_ids:
            if cid not in self.counts:
                self.counts[cid] = 0
                self.values[cid] = 0.0
                self.client_losses[cid] = 0.0
                self.client_sizes[cid]  = 1
                self._has_loss[cid]     = False

    def register_sizes(self, clients):
        """Cache the number of training samples per client (one-time)."""
        for c in clients:
            cid = c.client_id
            self.client_sizes[cid] = max(1, getattr(c, "n_train", 1))

    def select(self, clients, k):
        self.t += 1
        if k >= len(clients):
            return clients
        cids = [c.client_id for c in clients]
        self.register(cids)
        # Refresh sizes lazily (covers new clients joining mid-run)
        for c in clients:
            if self.client_sizes.get(c.client_id, 1) <= 1:
                self.client_sizes[c.client_id] = max(1, getattr(c, "n_train", 1))

        scores = {}
        for c in clients:
            cid = c.client_id
            n = self.counts[cid]
            if n == 0 or not self._has_loss[cid]:
                scores[cid] = float("inf")
            else:
                util = self.alpha * np.sqrt(self.client_sizes[cid]) * self.client_losses[cid]
                expl = self.c * np.sqrt(np.log(self.t) / n)
                scores[cid] = util + expl
        sorted_clients = sorted(clients, key=lambda c: -scores[c.client_id])
        return sorted_clients[:k]

    def update(self, selected_clients, reward: float, client_losses: dict = None, **kwargs):
        super().update(selected_clients, reward, **kwargs)
        if client_losses:
            for cid, loss in client_losses.items():
                if cid in self.client_losses:
                    self.client_losses[cid] = float(loss)
                    self._has_loss[cid] = True


# ─────────────────────────────────────────────────────────────────────────────
# LossBasedSelector  (simplified loss-greedy + UCB exploration)
# ─────────────────────────────────────────────────────────────────────────────

class LossBasedSelector(UCBClientSelector):
    """
    Pure loss-greedy selector with UCB1 exploration bonus.

    score_i(t) = loss_i + c * sqrt(log(t) / n_i)

    Differs from Oort in that it does NOT scale by sqrt(N_i): all clients
    are treated equally regardless of dataset size. Useful as a controlled
    contrast to isolate the effect of size weighting.
    """

    def __init__(self, client_ids, c: float = 2.0):
        super().__init__(client_ids, c=c)
        self.client_losses = {cid: 0.0 for cid in client_ids}
        self._has_loss     = {cid: False for cid in client_ids}

    def register(self, client_ids):
        for cid in client_ids:
            if cid not in self.counts:
                self.counts[cid] = 0
                self.values[cid] = 0.0
                self.client_losses[cid] = 0.0
                self._has_loss[cid]     = False

    def select(self, clients, k):
        self.t += 1
        if k >= len(clients):
            return clients
        cids = [c.client_id for c in clients]
        self.register(cids)

        scores = {}
        for c in clients:
            cid = c.client_id
            n = self.counts[cid]
            if n == 0 or not self._has_loss[cid]:
                scores[cid] = float("inf")
            else:
                scores[cid] = (self.client_losses[cid]
                               + self.c * np.sqrt(np.log(self.t) / n))
        sorted_clients = sorted(clients, key=lambda c: -scores[c.client_id])
        return sorted_clients[:k]

    def update(self, selected_clients, reward: float, client_losses: dict = None, **kwargs):
        super().update(selected_clients, reward, **kwargs)
        if client_losses:
            for cid, loss in client_losses.items():
                if cid in self.client_losses:
                    self.client_losses[cid] = float(loss)
                    self._has_loss[cid] = True


# ─────────────────────────────────────────────────────────────────────────────
# FedCorSelector  (gradient-correlation approximation; Tang et al. NeurIPS 2022)
# ─────────────────────────────────────────────────────────────────────────────

class FedCorSelector(UCBClientSelector):
    """
    FedCor-style selector (simplified): prioritises clients whose recent
    weight delta has large magnitude (proxy for gradient informativeness)
    while penalising clients whose deltas are highly correlated with the
    aggregated global update (redundant contributions).

    score_i(t) = eta * ||Δw_i||  -  gamma * cos(Δw_i, Δw_global)
                + c * sqrt(log(t) / n_i)

    Implementation note: stores the L2 norm and a one-dimensional
    cosine-similarity scalar per client to keep memory O(|clients|).
    """

    def __init__(self, client_ids, c: float = 2.0,
                 eta: float = 1.0, gamma: float = 0.5):
        super().__init__(client_ids, c=c)
        self.eta   = eta
        self.gamma = gamma
        self.client_delta_norms = {cid: 0.0 for cid in client_ids}
        self.client_delta_cos   = {cid: 0.0 for cid in client_ids}
        self._has_delta         = {cid: False for cid in client_ids}

    def register(self, client_ids):
        for cid in client_ids:
            if cid not in self.counts:
                self.counts[cid] = 0
                self.values[cid] = 0.0
                self.client_delta_norms[cid] = 0.0
                self.client_delta_cos[cid]   = 0.0
                self._has_delta[cid]         = False

    def select(self, clients, k):
        self.t += 1
        if k >= len(clients):
            return clients
        cids = [c.client_id for c in clients]
        self.register(cids)

        scores = {}
        for c in clients:
            cid = c.client_id
            n = self.counts[cid]
            if n == 0 or not self._has_delta[cid]:
                scores[cid] = float("inf")
            else:
                util = (self.eta   * self.client_delta_norms[cid]
                        - self.gamma * self.client_delta_cos[cid])
                expl = self.c * np.sqrt(np.log(self.t) / n)
                scores[cid] = util + expl
        sorted_clients = sorted(clients, key=lambda c: -scores[c.client_id])
        return sorted_clients[:k]

    def update(self, selected_clients, reward: float,
               client_delta_norms: dict = None,
               client_delta_cosines: dict = None,
               **kwargs):
        super().update(selected_clients, reward, **kwargs)
        if client_delta_norms:
            for cid, val in client_delta_norms.items():
                if cid in self.client_delta_norms:
                    self.client_delta_norms[cid] = float(val)
                    self._has_delta[cid] = True
        if client_delta_cosines:
            for cid, val in client_delta_cosines.items():
                if cid in self.client_delta_cos:
                    self.client_delta_cos[cid] = float(val)
