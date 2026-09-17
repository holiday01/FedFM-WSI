"""
Baseline comparisons for FL paper:
  1. Local-only  — each client trains independently, no federation
  2. Centralized — all data pooled at one server (upper bound)
  3. FedAvg      — federation (already in fl_agent.py)

Generates 3-way comparison: Local < FL < Centralized
"""

import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from fl_agent.federated_learning import make_dataloader
from models.classifiers import build_model


def train_model(model, train_samples, cfg, epochs=None):
    """Train a model on a sample list for N epochs."""
    device = cfg["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["federated"]["local_lr"])
    criterion = nn.CrossEntropyLoss()
    loader = make_dataloader(train_samples, cfg["experiment"]["batch_size"], True, device)
    if loader is None:
        return model
    n_epochs = epochs or (cfg["federated"]["num_rounds"] * cfg["federated"]["local_epochs"])
    for _ in range(n_epochs):
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model


def eval_model(model, test_samples, cfg) -> float:
    device = cfg["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.eval()
    model.to(device)
    loader = make_dataloader(test_samples, 64, False, device)
    if loader is None:
        return 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
    return correct / total if total > 0 else 0.0


def run_local_only(splits: dict, global_test: list, cfg: dict, task: str) -> dict:
    """
    Local-only baseline: each client trains independently on its own data.
    Evaluation: each client's model is tested on the global test set.
    Returns: {per_client_acc, mean_acc, weighted_acc}
    """
    print("[Baseline] Running local-only training...")
    input_dim = cfg["model_dims"][cfg.get("_current_model", "UNI_v2")]
    device = cfg["experiment"]["device"]

    per_client_accs = {}
    all_preds = []
    for cid, data in splits.items():
        if not data["train"]:
            continue
        model = build_model(task, input_dim, cfg)
        model = train_model(model, data["train"], cfg)
        acc = eval_model(model, global_test, cfg)
        per_client_accs[cid] = {"acc": acc, "n_train": len(data["train"])}

    accs = [v["acc"] for v in per_client_accs.values()]
    ns = [v["n_train"] for v in per_client_accs.values()]
    mean_acc = float(np.mean(accs))
    weighted_acc = float(np.average(accs, weights=ns))

    print(f"  Local-only: mean={mean_acc:.4f}  weighted={weighted_acc:.4f}")
    return {
        "mean_accuracy": mean_acc,
        "weighted_accuracy": weighted_acc,
        "per_client": per_client_accs,
    }


def run_centralized(splits: dict, global_test: list, cfg: dict, task: str) -> dict:
    """Pool all training data and train a single model."""
    print("[Baseline] Running centralized training...")
    all_train = [s for data in splits.values() for s in data["train"]]
    input_dim = cfg["model_dims"][cfg.get("_current_model", "UNI_v2")]
    model = build_model(task, input_dim, cfg)
    model = train_model(model, all_train, cfg)
    acc = eval_model(model, global_test, cfg)
    print(f"  Centralized: acc={acc:.4f}  (n_train={len(all_train)})")
    return {"accuracy": acc, "n_train": len(all_train)}
