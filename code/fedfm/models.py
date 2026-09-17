"""Heads and losses. Architectures are identical to legacy_copy/models/classifiers.py."""
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(512, 256), dropout=0.3, batch_norm=False):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def bn_keys(self):
        keys = set()
        for name, mod in self.named_modules():
            if isinstance(mod, nn.BatchNorm1d):
                for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    keys.add(f"{name}.{p}")
        return keys


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def bn_keys(self):
        return set()


class SurvivalMLP(MLPClassifier):
    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.3):
        super().__init__(input_dim, 1, hidden_dims, dropout, batch_norm=False)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _risk_set_logsumexp(risk, time, strata=None):
    """log sum_{j in R(t_i)} exp(risk_j) for every i, with R(t_i) = {j : t_j >= t_i} (tied times
    share one risk set, Breslow) restricted to the same stratum when `strata` is given.
    Fully vectorised (O(n^2) memory, n = number of patients), exact and permutation invariant."""
    comp = time[None, :] >= time[:, None]
    if strata is not None:
        comp = comp & (strata[None, :] == strata[:, None])
    logits = risk[None, :].expand(risk.shape[0], -1).masked_fill(~comp, float("-inf"))
    return torch.logsumexp(logits, dim=1)


def cox_loss(risk, time, event):
    """Negative Breslow partial log-likelihood, normalised by the number of samples.
    Tied times share the full risk set {j : t_j >= t_i}; the value is invariant to the
    order of the samples.  (The legacy implementation used a sequential cumulative
    sum, which excluded tied samples from each other's risk sets.)"""
    lse = _risk_set_logsumexp(risk, time)
    return -torch.sum((risk - lse) * event) / risk.shape[0]


def stratified_cox_loss(risk, time, event, strata):
    """Site-stratified Breslow partial likelihood: risk sets restricted to the same stratum,
    normalised by the total number of samples (= size-weighted sum of per-stratum losses)."""
    lse = _risk_set_logsumexp(risk, time, strata)
    return -torch.sum((risk - lse) * event) / risk.shape[0]


def cox_loss_legacy(risk, time, event):
    """The legacy implementation (sequential cumulative sum after a descending-time sort):
    tied samples are excluded from each other's risk sets and the value depends on the
    order of tied samples.  Kept ONLY to reproduce the legacy survival numbers."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cumsum) * event)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
