"""
AutoFL-WSI Agent
================
Autonomous agent that automatically:
1. Scans foundation model features
2. Partitions by TCGA Tissue Source Site (simulated federation)
3. Builds federated learning clients
4. Trains with FedAvg (or other algorithm)
5. Evaluates and saves results

Usage:
    python fl_agent.py --model UNI_v2 --task cancer_type_classification
    python fl_agent.py --model Virchow2 --algo FedProx --rounds 30
    python fl_agent.py --all-models  # run all foundation models sequentially
"""

import os
import sys
import json
import yaml
import copy
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.partition import TCGAFLPartitioner, TCGASurvivalPartitioner
from models.classifiers import build_model, concordance_index
from fl_agent.federated_learning import (FLClient, FLServer, make_dataloader,
                                          make_survival_dataloader)
from fl_agent.bandit_selection import (UCBClientSelector, PathologyAwareSelector,
                                        OortSelector, LossBasedSelector,
                                        FedCorSelector)

# ─────────────────────────────────────────────
# Config utilities
# ─────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "fl_config.yaml"


def load_config(path: str = None) -> dict:
    p = path or CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# AutoFL Agent
# ─────────────────────────────────────────────

class AutoFLAgent:
    """
    Autonomous FL pipeline agent for WSI foundation model features.
    One agent instance = one (foundation_model, task, algorithm) experiment.
    """

    def __init__(
        self,
        foundation_model: str = "UNI_v2",
        task: str = "cancer_type_classification",
        algorithm: str = "FedAvg",
        num_rounds: Optional[int] = None,
        clients_per_round: Optional[int] = None,
        cfg_path: Optional[str] = None,
        results_dir: Optional[str] = None,
    ):
        self.cfg = load_config(cfg_path)
        self.foundation_model = foundation_model
        self.task = task

        # Override config with runtime args
        if algorithm:
            self.cfg["federated"]["algorithm"] = algorithm
        if num_rounds:
            self.cfg["federated"]["num_rounds"] = num_rounds
        if clients_per_round:
            self.cfg["federated"]["clients_per_round"] = clients_per_round
        # True FedBN: enable BatchNorm1d layers so BN params actually exist.
        # The server aggregates ONLY non-BN parameters; each client retains its
        # own BN state (running stats + affine weights) between rounds and uses
        # it for local inference — the correct per-client evaluation protocol.
        if self.cfg["federated"]["algorithm"] == "FedBN":
            self.cfg["model"]["batch_norm"] = True

        self.results_dir = Path(results_dir or self.cfg["data"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = f"{foundation_model}_{task}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = self.cfg["experiment"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[Agent] CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.cfg["experiment"]["device"] = "cpu"

        print(f"\n{'='*60}")
        print(f"  AutoFL-WSI Agent")
        print(f"  Model  : {foundation_model}")
        print(f"  Task   : {task}")
        print(f"  Algo   : {self.cfg['federated']['algorithm']}")
        print(f"  Rounds : {self.cfg['federated']['num_rounds']}")
        print(f"  Device : {self.device}")
        print(f"  RunID  : {self.run_id}")
        print(f"{'='*60}\n")

    # ── Step 1: Partition ────────────────────────────────────────

    def partition_data(self) -> tuple:
        """Scan and partition TCGA WSI features by TSS."""
        print("[Agent Step 1] Partitioning data by TCGA Tissue Source Site...")
        partitioner = TCGAFLPartitioner(
            feature_dir=self.cfg["data"]["feature_dir"],
            model_name=self.foundation_model,
            min_samples=self.cfg["federated"]["min_samples_per_client"],
            seed=self.cfg["experiment"]["seed"],
            train_ratio=self.cfg["experiment"]["train_ratio"],
            val_ratio=self.cfg["experiment"]["val_ratio"],
        )
        partitioner.scan_features()
        splits = partitioner.split_clients()
        global_test = partitioner.build_global_test_set(splits)
        stats = partitioner.summary()

        # Save partition manifest
        manifest_path = self.run_dir / "partition_manifest.json"
        partitioner.save_partition(str(manifest_path))

        print(f"[Agent Step 1] {stats['num_clients']} FL clients, {stats['total_samples']} total samples")
        return splits, global_test, stats

    # ── Step 2: Build model ──────────────────────────────────────

    def build_global_model(self) -> torch.nn.Module:
        """Initialize the global model."""
        input_dim = self.cfg["model_dims"][self.foundation_model]
        model = build_model(self.task, input_dim, self.cfg)
        model.to(self.device)
        print(f"[Agent Step 2] Global model: {type(model).__name__}  input_dim={input_dim}")
        return model

    # ── Step 3: Build FL clients ─────────────────────────────────

    def build_clients(self, splits: dict, global_model: torch.nn.Module) -> List[FLClient]:
        """Instantiate FL clients from partition splits."""
        clients = []
        for cid, data in splits.items():
            if not data["train"]:
                continue
            client = FLClient(
                client_id=cid,
                train_samples=data["train"],
                val_samples=data["val"],
                test_samples=data.get("test", []),
                model=global_model,
                cfg=self.cfg,
            )
            clients.append(client)
        print(f"[Agent Step 3] Built {len(clients)} FL clients")
        return clients

    # ── Step 4: Train ────────────────────────────────────────────

    def _compute_class_weights(self, splits: dict) -> torch.Tensor:
        """Compute inverse-frequency class weights from training set."""
        from collections import Counter
        counts = Counter()
        for data in splits.values():
            for s in data["train"]:
                counts[s["cancer_label"]] += 1
        n_classes = self.cfg["tasks"]["cancer_type_classification"]["num_classes"]
        total = sum(counts.values())
        weights = []
        for c in range(n_classes):
            w = total / (n_classes * counts[c]) if counts[c] > 0 else 1.0
            weights.append(w)
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"[Agent] Class weights: {[f'{w:.2f}' for w in weights]}")
        return w_tensor

    def train(self, clients: List[FLClient], global_model: torch.nn.Module,
              splits: dict = None, global_test: list = None) -> tuple:
        """Run federated training with class-weighted loss."""
        class_weights = None
        if self.task == "cancer_type_classification" and splits is not None:
            class_weights = self._compute_class_weights(splits)

        # Build client selector based on config
        algo     = self.cfg["federated"]["algorithm"]
        selector_type = self.cfg["federated"].get("client_selector", "stratified")
        client_selector = None

        if selector_type == "ucb":
            client_selector = UCBClientSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
            )
        elif selector_type == "pathology_aware":
            pa_sel = PathologyAwareSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                beta=self.cfg["federated"].get("pa_beta", 1.0),
                lam=self.cfg["federated"].get("pa_lambda", 0.5),
                rho=self.cfg["federated"].get("pa_rho", 0.2),
                window=self.cfg["federated"].get("pa_window", 10),
            )
            print("[Agent] Pre-computing client feature means for PathologyAwareSelector...")
            pa_sel.register_feature_means(clients)
            client_selector = pa_sel
        elif selector_type == "oort":
            oort_sel = OortSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                alpha=self.cfg["federated"].get("oort_alpha", 1.0),
            )
            oort_sel.register_sizes(clients)
            client_selector = oort_sel
        elif selector_type == "loss_based":
            client_selector = LossBasedSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
            )
        elif selector_type == "fedcor":
            client_selector = FedCorSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                eta=self.cfg["federated"].get("fedcor_eta", 1.0),
                gamma=self.cfg["federated"].get("fedcor_gamma", 0.5),
            )

        server = FLServer(
            global_model, self.cfg,
            class_weights=class_weights,
            client_selector=client_selector,
            n_total_clients=len(clients),
        )
        # Pool per-client validation splits for early-stopping monitoring.
        # The held-out test set (global_test) is kept strictly for final evaluation.
        global_val = [s for data in splits.values() for s in data["val"]] if splits else None
        history = server.train(clients, task=self.task, global_test=global_val)
        return server, history

    # ── Step 5: Evaluate ─────────────────────────────────────────

    def evaluate(self, server: FLServer, global_test: list, splits: dict,
                clients: List[FLClient] = None) -> dict:
        """Evaluate global model on global test set."""
        print("\n[Agent Step 5] Evaluating global model on held-out test set...")
        algo = self.cfg["federated"]["algorithm"]
        if algo == "FedBN" and clients is not None:
            # Per-client evaluation with local BN state
            global_result = server.global_evaluate_fedbn(clients, device=self.device)
            return global_result
        global_result = server.global_evaluate(global_test, device=self.device)

        # Per-project accuracy
        project_results = {}
        projects = list(set(s["project"] for s in global_test))
        for proj in projects:
            proj_samples = [s for s in global_test if s["project"] == proj]
            loader = make_dataloader(proj_samples, batch_size=64, shuffle=False, device=self.device)
            if loader is None:
                continue
            server.global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in loader:
                    preds = server.global_model(X).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += X.size(0)
            project_results[proj] = {
                "accuracy": correct / total if total > 0 else 0.0,
                "n": total
            }
            print(f"  {proj}: acc={project_results[proj]['accuracy']:.4f}  (n={total})")

        return {**global_result, "per_project": project_results}

    # ── Step 6: Baseline comparison ──────────────────────────────

    def run_centralized_baseline(self, splits: dict, global_test: list) -> dict:
        """Centralized training baseline (all data pooled)."""
        print("\n[Agent Baseline] Centralized training...")
        all_train = []
        for data in splits.values():
            all_train.extend(data["train"])

        input_dim = self.cfg["model_dims"][self.foundation_model]
        model = build_model(self.task, input_dim, self.cfg).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        loader = make_dataloader(all_train, batch_size=self.cfg["experiment"]["batch_size"],
                                  shuffle=True, device=self.device)
        if loader is None:
            return {}

        total_epochs = self.cfg["federated"].get("centralized_epochs", 300)
        model.train()
        for epoch in range(total_epochs):
            for X, y in loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        loader_test = make_dataloader(global_test, batch_size=64, shuffle=False, device=self.device)
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader_test:
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += X.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"  Centralized baseline accuracy: {acc:.4f}  (n={total})")
        return {"accuracy": acc, "n": total}

    # ── Main orchestration ────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full AutoFL-WSI pipeline."""
        t_start = time.time()

        # Step 1: Partition
        splits, global_test, stats = self.partition_data()

        # Step 2: Build model
        global_model = self.build_global_model()

        # Step 3: Build clients
        clients = self.build_clients(splits, global_model)

        # Step 4: Train
        server, history = self.train(clients, global_model, splits=splits, global_test=global_test)

        # Step 5: Evaluate (FedBN uses per-client evaluation with local BN states)
        eval_result = self.evaluate(server, global_test, splits, clients=clients)

        # Step 6: Centralized baseline
        centralized = self.run_centralized_baseline(splits, global_test)

        elapsed = time.time() - t_start

        # Save everything
        summary = {
            "run_id": self.run_id,
            "foundation_model": self.foundation_model,
            "task": self.task,
            "algorithm": self.cfg["federated"]["algorithm"],
            "num_rounds": self.cfg["federated"]["num_rounds"],
            "num_clients": stats["num_clients"],
            "total_samples": stats["total_samples"],
            "federated_test_accuracy": eval_result.get("test_accuracy"),
            "centralized_test_accuracy": centralized.get("accuracy"),
            "per_project_accuracy": eval_result.get("per_project"),
            "training_history": history,
            "elapsed_seconds": elapsed,
        }

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Agent] Results saved to {self.run_dir}")
        print(f"[Agent] Total time: {elapsed/60:.1f} min")
        print(f"  FL accuracy     : {summary['federated_test_accuracy']:.4f}")
        print(f"  Central baseline: {summary['centralized_test_accuracy']:.4f}")

        # Save model checkpoint
        ckpt_path = self.run_dir / "global_model.pt"
        torch.save(server.global_model.state_dict(), ckpt_path)

        # Save adaptive-selector state for E7 interpretability analysis.
        # Only saved when an adaptive selector was used (UCB/PathologyAware/Oort/etc.).
        if server.client_selector is not None:
            sel = server.client_selector
            sel_state = {
                "selector_type": type(sel).__name__,
                "counts": dict(getattr(sel, "counts", {})),
                "values": dict(getattr(sel, "values", {})),
            }
            for attr in ("client_losses", "client_delta_norms",
                         "client_delta_cos", "cancer_map"):
                if hasattr(sel, attr):
                    sel_state[attr] = dict(getattr(sel, attr))
            with open(self.run_dir / "selector_state.json", "w") as f:
                json.dump(sel_state, f, indent=2, default=str)

        return summary


# ─────────────────────────────────────────────
# Multi-model sweep
# ─────────────────────────────────────────────

def run_all_models(cfg: dict, task: str = "cancer_type_classification", algo: str = "FedAvg"):
    """Run FL experiment for all foundation models and aggregate results."""
    models = cfg["foundation_models"]
    all_results = []
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# Running: {model_name}")
        print(f"{'#'*60}")
        agent = AutoFLAgent(
            foundation_model=model_name,
            task=task,
            algorithm=algo,
        )
        result = agent.run()
        all_results.append({
            "model": model_name,
            "fl_accuracy": result["federated_test_accuracy"],
            "centralized_accuracy": result["centralized_test_accuracy"],
        })

    df = pd.DataFrame(all_results)
    results_dir = Path(cfg["data"]["results_dir"])
    out_path = results_dir / f"all_models_{task}_{algo}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Sweep] All-model results saved to {out_path}")
    print(df.to_string())
    return df


# ─────────────────────────────────────────────
# Survival FL Agent
# ─────────────────────────────────────────────

class SurvivalFLAgent:
    """
    FL agent for survival prediction (OS time + event).
    Uses TCGASurvivalPartitioner (BRCA, COAD, STAD only).
    Trains SurvivalMLP with Cox partial log-likelihood loss.
    Evaluates with Concordance Index (C-index).
    """

    def __init__(self, foundation_model: str = "UNI_v2",
                 algorithm: str = "FedAvg", cfg_path: str = None):
        self.cfg = load_config(cfg_path)
        self.foundation_model = foundation_model
        self.cfg["federated"]["algorithm"] = algorithm
        self.device = self.cfg["experiment"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.cfg["experiment"]["device"] = "cpu"

        self.results_dir = Path(self.cfg["data"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = (f"{foundation_model}_survival_{algorithm}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  SurvivalFL Agent  model={foundation_model}  algo={algorithm}")
        print(f"{'='*60}\n")

    def run(self) -> dict:
        import copy
        t_start = time.time()

        # Partition
        partitioner = TCGASurvivalPartitioner(
            feature_dir=self.cfg["data"]["feature_dir"],
            model_name=self.foundation_model,
            clinical_csv=self.cfg["data"]["clinical_csv"],
            min_samples=5,
            seed=self.cfg["experiment"]["seed"],
            train_ratio=self.cfg["experiment"]["train_ratio"],
            val_ratio=self.cfg["experiment"]["val_ratio"],
        )
        partitioner.scan_features()

        # Optional per-cancer isolation for E5 (consumed here, set by
        # run_e5_survival.py). Without this, BRCA/COAD/STAD all train on
        # the identical 3-cancer pool and produce duplicate rows.
        cancer_filter = self.cfg["data"].get("survival_cancer_filter")
        if cancer_filter:
            project = (cancer_filter if cancer_filter.startswith("TCGA-")
                       else f"TCGA-{cancer_filter}")
            partitioner = partitioner.filter_by_project(project)
            print(f"[SurvivalPartitioner] Filtered to {project}: "
                  f"{len(partitioner.clients)} clients")

        splits = partitioner.split_clients()
        global_test = partitioner.build_global_test_set(splits)
        stats = partitioner.summary()
        print(f"Survival: {stats['num_clients']} clients, {stats['total_samples']} samples")

        # Build model
        input_dim = self.cfg["model_dims"][self.foundation_model]
        global_model = build_model("survival_prediction", input_dim, self.cfg)
        global_model.to(self.device)

        # Build clients
        clients = []
        for cid, data in splits.items():
            if not data["train"]:
                continue
            client = FLClient(
                client_id=cid,
                train_samples=data["train"],
                val_samples=data["val"],
                model=copy.deepcopy(global_model),
                cfg=self.cfg,
            )
            clients.append(client)
        print(f"Built {len(clients)} FL clients for survival")

        # Build client selector from config (same vocabulary as AutoFLAgent)
        selector_type = self.cfg["federated"].get("client_selector", "stratified")
        client_selector = None
        if selector_type == "ucb":
            client_selector = UCBClientSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
            )
        elif selector_type == "pathology_aware":
            pa_sel = PathologyAwareSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                beta=self.cfg["federated"].get("pa_beta", 1.0),
                lam=self.cfg["federated"].get("pa_lambda", 0.5),
                rho=self.cfg["federated"].get("pa_rho", 0.2),
                window=self.cfg["federated"].get("pa_window", 10),
            )
            pa_sel.register_feature_means(clients)
            client_selector = pa_sel
        elif selector_type == "oort":
            oort_sel = OortSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                alpha=self.cfg["federated"].get("oort_alpha", 1.0),
            )
            oort_sel.register_sizes(clients)
            client_selector = oort_sel
        elif selector_type == "loss_based":
            client_selector = LossBasedSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
            )
        elif selector_type == "fedcor":
            client_selector = FedCorSelector(
                [c.client_id for c in clients],
                c=self.cfg["federated"].get("ucb_c", 2.0),
                eta=self.cfg["federated"].get("fedcor_eta", 1.0),
                gamma=self.cfg["federated"].get("fedcor_gamma", 0.5),
            )

        # Train (no class weights for survival)
        server = FLServer(global_model, self.cfg, class_weights=None,
                          client_selector=client_selector,
                          n_total_clients=len(clients))
        server.train(clients, task="survival_prediction", global_test=None)

        # Evaluate FL model (C-index)
        fl_result = server.global_evaluate_survival(global_test, device=self.device)

        # Centralized baseline
        all_train = [s for d in splits.values() for s in d["train"]]
        central_model = build_model("survival_prediction", input_dim, self.cfg)
        central_model.to(self.device)
        optimizer = torch.optim.Adam(central_model.parameters(), lr=1e-3)
        from models.classifiers import cox_loss
        loader = make_survival_dataloader(all_train, 32, True, self.device)
        total_epochs = self.cfg["federated"].get("centralized_epochs", 300)
        if loader:
            central_model.train()
            for _ in range(total_epochs):
                for X, t, e in loader:
                    optimizer.zero_grad()
                    loss = cox_loss(central_model(X), t, e)
                    if not torch.isnan(loss):
                        loss.backward()
                        optimizer.step()
        central_model.eval()
        ct_loader = make_survival_dataloader(global_test, 64, False, self.device)
        all_risk, all_t, all_e = [], [], []
        if ct_loader:
            with torch.no_grad():
                for X, t, e in ct_loader:
                    all_risk.extend(central_model(X).cpu().numpy())
                    all_t.extend(t.cpu().numpy())
                    all_e.extend(e.cpu().numpy())
        central_cindex = concordance_index(all_risk, all_t, all_e) if all_risk else 0.5
        print(f"  Centralized C-index: {central_cindex:.4f}")

        elapsed = time.time() - t_start
        summary = {
            "run_id": self.run_id,
            "foundation_model": self.foundation_model,
            "task": "survival_prediction",
            "algorithm": self.cfg["federated"]["algorithm"],
            "num_rounds": self.cfg["federated"]["num_rounds"],
            "num_clients": stats["num_clients"],
            "fl_cindex":   fl_result.get("c_index", 0.5),
            "central_cindex": central_cindex,
            "n_test": fl_result.get("n_test", 0),
            "elapsed_seconds": elapsed,
        }
        with open(self.run_dir / "summary_survival.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  FL C-index     : {summary['fl_cindex']:.4f}")
        print(f"  Central C-index: {summary['central_cindex']:.4f}")
        return summary


def run_survival_all_models(cfg: dict, algo: str = "FedAvg") -> pd.DataFrame:
    """Run survival FL for all foundation models."""
    models = cfg["foundation_models"]
    rows = []
    for m in models:
        agent = SurvivalFLAgent(foundation_model=m, algorithm=algo)
        r = agent.run()
        rows.append({"model": m, "fl_cindex": r["fl_cindex"],
                     "central_cindex": r["central_cindex"]})
    df = pd.DataFrame(rows)
    out = Path(cfg["data"]["results_dir"]) / f"survival_all_models_{algo}.csv"
    df.to_csv(out, index=False)
    print(df.to_string())
    return df


def run_survival_within_cancer(cfg: dict, algo: str = "FedAvg") -> pd.DataFrame:
    """
    Within-cancer survival FL: BRCA, COAD, STAD each run separately for all 7 FMs.
    Fixes cross-cancer Cox gradient conflict (C-index < 0.5 issue).
    Full-batch Cox loss per client.
    """
    from data.partition import TCGASurvivalPartitioner
    from models.classifiers import build_model, cox_loss, concordance_index
    from fl_agent.federated_learning import FLClient, FLServer, make_survival_dataloader

    projects = ["TCGA-BRCA", "TCGA-COAD", "TCGA-STAD"]
    models_list = cfg["foundation_models"]
    results_dir = Path(cfg["data"]["results_dir"])
    rows = []

    for model_name in models_list:
        print(f"\n{'='*60}\n  Within-Cancer Survival: {model_name}\n{'='*60}")

        partitioner = TCGASurvivalPartitioner(
            feature_dir=cfg["data"]["feature_dir"],
            model_name=model_name,
            clinical_csv=cfg["data"]["clinical_csv"],
            min_samples=5,
            seed=cfg["experiment"]["seed"],
            train_ratio=cfg["experiment"]["train_ratio"],
            val_ratio=cfg["experiment"]["val_ratio"],
        )
        partitioner.scan_features()

        for proj in projects:
            sub = partitioner.filter_by_project(proj)
            if len(sub.clients) < 2:
                print(f"  [{proj}] only {len(sub.clients)} client(s) — skip")
                continue

            splits = sub.split_clients()
            global_test = sub.build_global_test_set(splits)
            n_clients = len(splits)
            print(f"  [{proj}] {n_clients} clients, "
                  f"{sum(len(v) for v in sub.clients.values())} samples")

            device = cfg["experiment"]["device"]
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            input_dim = cfg["model_dims"][model_name]
            global_model = build_model("survival_prediction", input_dim, cfg)
            global_model.to(device)

            clients = [
                FLClient(cid, data["train"], data["val"],
                         copy.deepcopy(global_model), cfg)
                for cid, data in splits.items() if data["train"]
            ]

            server = FLServer(global_model, cfg, class_weights=None)
            server.train(clients, task="survival_prediction", global_test=None)

            fl_result = server.global_evaluate_survival(global_test, device=device)
            fl_ci = fl_result.get("c_index", 0.5)

            # Centralized baseline (full batch)
            all_train = [s for d in splits.values() for s in d["train"]]
            c_model = build_model("survival_prediction", input_dim, cfg)
            c_model.to(device)
            opt = torch.optim.Adam(c_model.parameters(), lr=1e-3)
            loader = make_survival_dataloader(all_train, len(all_train), False, device)
            total_ep = cfg["federated"].get("centralized_epochs", 300)
            if loader:
                c_model.train()
                for _ in range(total_ep):
                    for X, t, e in loader:
                        opt.zero_grad()
                        loss = cox_loss(c_model(X), t, e)
                        if not torch.isnan(loss):
                            loss.backward()
                            opt.step()
            c_model.eval()
            ct_loader = make_survival_dataloader(global_test, len(global_test), False, device)
            r_all, t_all, e_all = [], [], []
            if ct_loader:
                with torch.no_grad():
                    for X, t, e in ct_loader:
                        r_all.extend(c_model(X).cpu().numpy())
                        t_all.extend(t.cpu().numpy())
                        e_all.extend(e.cpu().numpy())
            central_ci = concordance_index(r_all, t_all, e_all) if r_all else 0.5

            print(f"  [{proj}] FL={fl_ci:.4f}  Central={central_ci:.4f}")
            rows.append({
                "model": model_name,
                "cancer": proj.replace("TCGA-", ""),
                "fl_cindex": fl_ci,
                "central_cindex": central_ci,
                "n_clients": n_clients,
                "n_test": len(global_test),
            })

    df = pd.DataFrame(rows)
    out = results_dir / f"exp6_survival_within_cancer_{algo}.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string())
    return df


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoFL-WSI Agent")
    parser.add_argument("--model", default="UNI_v2",
                        choices=["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15",
                                 "CTransPath", "Midnight12k", "ResNet50"])
    parser.add_argument("--task", default="cancer_type_classification",
                        choices=["cancer_type_classification", "survival_prediction"])
    parser.add_argument("--algo", default="FedAvg",
                        choices=["FedAvg", "FedProx", "SCAFFOLD", "FedBN"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--all-models", action="store_true",
                        help="Run all foundation models sequentially")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.all_models:
        run_all_models(cfg, task=args.task, algo=args.algo)
    else:
        agent = AutoFLAgent(
            foundation_model=args.model,
            task=args.task,
            algorithm=args.algo,
            num_rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            cfg_path=args.config,
        )
        agent.run()
