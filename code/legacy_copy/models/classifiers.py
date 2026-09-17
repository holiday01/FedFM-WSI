"""
MLP classifier and survival prediction heads for WSI FL experiments.
Designed to work on top of frozen foundation model features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLPClassifier(nn.Module):
    """
    MLP classification head on top of foundation model features.
    Used for cancer type classification (9-class).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        self.load_state_dict(weights)

    def get_bn_param_names(self) -> set:
        """Return state_dict key names belonging to BatchNorm1d layers (all params)."""
        bn_names = set()
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                prefix = name + "." if name else ""
                for pname in ("weight", "bias", "running_mean",
                               "running_var", "num_batches_tracked"):
                    bn_names.add(prefix + pname)
        return bn_names


class SurvivalMLP(nn.Module):
    """
    MLP for survival prediction (Cox proportional hazards).
    Outputs a single risk score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        self.load_state_dict(weights)

    def get_bn_param_names(self) -> set:
        """Return state_dict key names belonging to BatchNorm1d layers (all params)."""
        bn_names = set()
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                prefix = name + "." if name else ""
                for pname in ("weight", "bias", "running_mean",
                               "running_var", "num_batches_tracked"):
                    bn_names.add(prefix + pname)
        return bn_names


def cox_loss(risk_scores: torch.Tensor, survival_time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Negative partial log-likelihood for Cox PH model.
    Args:
        risk_scores: (N,) predicted risk
        survival_time: (N,) observed times
        event: (N,) event indicator (1=event, 0=censored)
    """
    # Sort by survival time descending
    order = torch.argsort(survival_time, descending=True)
    risk_scores = risk_scores[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    loss = -torch.mean((risk_scores - log_cumsum) * event)
    return loss


def concordance_index(risk_scores, survival_time, event):
    """Compute Harrell's C-index."""
    import numpy as np
    risk = np.array(risk_scores)
    time = np.array(survival_time)
    evt = np.array(event, dtype=bool)

    concordant = 0
    permissible = 0
    for i in range(len(time)):
        if not evt[i]:
            continue
        for j in range(len(time)):
            if time[j] > time[i]:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


class LinearProbe(nn.Module):
    """
    Single linear layer — standard baseline for foundation model evaluation.
    Equivalent to logistic regression. Most commonly used in FM benchmark papers.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        self.load_state_dict(weights)


class LinearSurvivalProbe(nn.Module):
    """Linear Cox model — simplest survival baseline."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

    def get_weights(self) -> dict:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        self.load_state_dict(weights)


def build_model(task: str, input_dim: int, cfg: dict, arch: str = "mlp") -> nn.Module:
    """
    Build classification or survival model.
    arch: 'mlp' (default, 2-hidden-layer MLP) | 'linear' (linear probe)
    """
    hidden_dims = cfg["model"]["hidden_dims"]
    dropout = cfg["model"]["dropout"]
    batch_norm = cfg["model"]["batch_norm"]
    if task == "cancer_type_classification":
        num_classes = cfg["tasks"]["cancer_type_classification"]["num_classes"]
        if arch == "linear":
            return LinearProbe(input_dim, num_classes)
        return MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        )
    elif task == "survival_prediction":
        if arch == "linear":
            return LinearSurvivalProbe(input_dim)
        return SurvivalMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        )
    else:
        raise ValueError(f"Unknown task: {task}")
