"""
FL Data Partitioner
Assigns TCGA WSI samples to federated learning clients
based on Tissue Source Site (TSS) codes from TCGA IDs.

TCGA ID format: TCGA-{PROJECT}-{TSS}-{PARTICIPANT}-...
FL client = PROJECT_TSS  (e.g., TCGA-BRCA_BH)
"""

import os
import re
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


CANCER_TYPE_MAP = {
    "TCGA-BRCA": 0, "TCGA-COAD": 1, "TCGA-STAD": 2, "TCGA-LGG": 3,
    "TCGA-LUAD": 4, "TCGA-HNSC": 5, "TCGA-SKCM": 6, "TCGA-CESC": 7, "TCGA-PAAD": 8,
}


WSI_IMAGE_ROOT = "/mnt/10t/holiday/1019_image"


def build_case_to_project_map(image_root: str = WSI_IMAGE_ROOT) -> Dict[str, str]:
    """
    Build case_id → project_id mapping by scanning the WSI directory structure.
    Directories: {image_root}/{TCGA-*}/{case}.svs
    This is more reliable than parsing TSS codes from filenames.
    """
    mapping = {}
    root = Path(image_root)
    for proj_dir in root.iterdir():
        if not proj_dir.is_dir() or not proj_dir.name.startswith("TCGA-"):
            continue
        proj = proj_dir.name
        for svs in proj_dir.glob("*.svs"):
            parts = svs.stem.split("-")
            if len(parts) >= 3:
                case = "-".join(parts[:3])  # TCGA-XX-YYYY
                mapping[case] = proj
    return mapping


def parse_tcga_id(filename: str, case_to_proj: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse TCGA filename to extract project, TSS, and case ID.
    Uses case_to_proj mapping (built from WSI directory structure) for project lookup.
    Returns (project, tss, case_id) or (None, None, None) on failure.
    """
    parts = filename.split("-")
    if len(parts) < 3 or parts[0] != "TCGA":
        return None, None, None
    tss = parts[1]
    case_id = "-".join(parts[:3])
    project = case_to_proj.get(case_id) if case_to_proj else None
    return project, tss, case_id


class TCGAFLPartitioner:
    """
    Partitions TCGA WSI features into federated learning clients.
    Each client = one Tissue Source Site (TSS) within a project.
    """

    def __init__(
        self,
        feature_dir: str,
        model_name: str = "UNI_v2",
        metadata_tsv: Optional[str] = None,
        min_samples: int = 10,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        self.feature_dir = Path(feature_dir) / model_name
        self.model_name = model_name
        self.metadata_tsv = metadata_tsv
        self.min_samples = min_samples
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

        self.clients: Dict[str, List[dict]] = defaultdict(list)  # client_id -> sample list
        self.global_test: List[dict] = []
        self._case_to_proj: Dict[str, str] = {}

        np.random.seed(seed)

    def scan_features(self) -> int:
        """Scan feature directory and assign samples to FL clients."""
        # Sort for reproducibility — glob order is filesystem-dependent
        npy_files = sorted(self.feature_dir.glob("*.npy"))
        print(f"[Partition] Found {len(npy_files)} feature files in {self.feature_dir}")

        # Build mapping once
        print("[Partition] Building case→project mapping from WSI directories...")
        self._case_to_proj = build_case_to_project_map()
        print(f"[Partition] Mapped {len(self._case_to_proj)} cases")

        for fp in npy_files:
            name = fp.stem.split(".")[0]  # strip UUID suffix
            project, tss, case_id = parse_tcga_id(name, self._case_to_proj)
            if project is None:
                continue
            client_id = f"{project}_{tss}"
            label = CANCER_TYPE_MAP.get(project, -1)
            self.clients[client_id].append({
                "path": str(fp),
                "case_id": case_id,
                "project": project,
                "tss": tss,
                "client_id": client_id,
                "cancer_label": label,
                "filename": fp.name,
            })

        # Filter clients with too few samples
        before = len(self.clients)
        self.clients = {k: v for k, v in self.clients.items() if len(v) >= self.min_samples}
        print(f"[Partition] {len(self.clients)}/{before} clients after min_samples={self.min_samples} filter")
        return len(self.clients)

    def split_clients(self) -> Dict[str, Dict[str, List[dict]]]:
        """
        Split each client's samples into train/val/test **by case_id**
        to prevent patient-level data leakage between splits.
        All slides from the same patient are assigned to the same split.
        """
        from collections import defaultdict
        splits = {}
        for client_id, samples in self.clients.items():
            # Group slides by case_id
            cases: Dict[str, list] = defaultdict(list)
            for s in samples:
                cases[s["case_id"]].append(s)

            case_ids = list(cases.keys())
            rng = np.random.default_rng(self.seed)
            rng.shuffle(case_ids)

            n_cases  = len(case_ids)
            n_train  = int(n_cases * self.train_ratio)
            n_val    = int(n_cases * self.val_ratio)

            train_slides = [s for cid in case_ids[:n_train]
                            for s in cases[cid]]
            val_slides   = [s for cid in case_ids[n_train:n_train + n_val]
                            for s in cases[cid]]
            test_slides  = [s for cid in case_ids[n_train + n_val:]
                            for s in cases[cid]]

            splits[client_id] = {
                "train": train_slides,
                "val":   val_slides,
                "test":  test_slides,
                "total": len(samples),
                "n_cases": n_cases,
            }
        return splits

    def build_global_test_set(self, splits: Dict) -> List[dict]:
        """Aggregate all client test sets into a global held-out test set."""
        global_test = []
        for client_data in splits.values():
            global_test.extend(client_data["test"])
        self.global_test = global_test
        return global_test

    def summary(self) -> dict:
        """Return partition statistics."""
        stats = {
            "model": self.model_name,
            "num_clients": len(self.clients),
            "total_samples": sum(len(v) for v in self.clients.values()),
            "clients": {},
        }
        for cid, samples in self.clients.items():
            project = samples[0]["project"]
            stats["clients"][cid] = {
                "n": len(samples),
                "project": project,
                "tss": samples[0]["tss"],
            }
        return stats

    def save_partition(self, out_path: str):
        """Save partition manifest as JSON."""
        splits = self.split_clients()
        manifest = {
            "model": self.model_name,
            "num_clients": len(self.clients),
            "splits": splits,
            "global_test": self.build_global_test_set(splits),
            "cancer_type_map": CANCER_TYPE_MAP,
        }
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[Partition] Saved manifest to {out_path}")
        return manifest


def load_partition(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Survival data loader
# ─────────────────────────────────────────────

SURVIVAL_PROJECTS = {"TCGA-BRCA", "TCGA-COAD", "TCGA-STAD"}


def load_survival_map(clinical_csv: str) -> Dict[str, dict]:
    """
    Load clinical CSV and build case_id → {os_time, os_status, project} map.
    Only keeps rows with valid OS_time and OS_status.
    """
    import pandas as pd
    df = pd.read_csv(clinical_csv)
    df = df.dropna(subset=["OS_time", "OS_status"])
    df = df[df["OS_time"] > 0]
    survival_map = {}
    for _, row in df.iterrows():
        survival_map[row["submitter_id"]] = {
            "os_time":   float(row["OS_time"]),
            "os_status": int(row["OS_status"]),
            "project":   str(row["Project"]),
        }
    return survival_map


class TCGASurvivalPartitioner:
    """
    Partitions TCGA WSI features into FL clients for survival prediction.
    Only uses BRCA, COAD, STAD (cases with clinical OS data).
    Client = TSS within a project (same as classification partitioner).
    """

    def __init__(
        self,
        feature_dir: str,
        model_name: str = "UNI_v2",
        clinical_csv: str = None,
        min_samples: int = 5,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        self.feature_dir = Path(feature_dir) / model_name
        self.model_name = model_name
        self.clinical_csv = clinical_csv
        self.min_samples = min_samples
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.clients: Dict[str, List[dict]] = defaultdict(list)
        np.random.seed(seed)

    def scan_features(self) -> int:
        """Match WSI features to clinical survival data and build clients."""
        if self.clinical_csv is None:
            raise ValueError("clinical_csv must be provided for survival task")

        print(f"[SurvivalPartitioner] Loading clinical data from {self.clinical_csv}")
        survival_map = load_survival_map(self.clinical_csv)
        print(f"[SurvivalPartitioner] {len(survival_map)} patients with valid survival data")

        print("[SurvivalPartitioner] Building case→project mapping...")
        case_to_proj = build_case_to_project_map()

        npy_files = list(self.feature_dir.glob("*.npy"))
        print(f"[SurvivalPartitioner] Scanning {len(npy_files)} feature files...")

        matched = 0
        for fp in npy_files:
            name = fp.stem.split(".")[0]
            parts = name.split("-")
            if len(parts) < 3 or parts[0] != "TCGA":
                continue
            case_id = "-".join(parts[:3])
            tss = parts[1]
            project = case_to_proj.get(case_id)
            if project not in SURVIVAL_PROJECTS:
                continue
            if case_id not in survival_map:
                continue
            surv = survival_map[case_id]
            client_id = f"{project}_{tss}"
            self.clients[client_id].append({
                "path":      str(fp),
                "case_id":   case_id,
                "project":   project,
                "tss":       tss,
                "client_id": client_id,
                "os_time":   surv["os_time"],
                "os_status": surv["os_status"],
                # cancer_label for stratification reference only
                "cancer_label": CANCER_TYPE_MAP.get(project, -1),
            })
            matched += 1

        before = len(self.clients)
        self.clients = {k: v for k, v in self.clients.items() if len(v) >= self.min_samples}
        print(f"[SurvivalPartitioner] Matched {matched} slides, "
              f"{len(self.clients)}/{before} clients after min_samples={self.min_samples} filter")
        return len(self.clients)

    def split_clients(self) -> Dict[str, Dict[str, List[dict]]]:
        """
        Split each client's samples into train/val/test **by case_id**
        to prevent patient-level data leakage between splits.
        All slides from the same patient are assigned to the same split.
        """
        from collections import defaultdict as _dd
        splits = {}
        for cid, samples in self.clients.items():
            # Group slides by case_id (patient)
            cases: dict = _dd(list)
            for s in samples:
                cases[s["case_id"]].append(s)

            case_ids = list(cases.keys())
            rng = np.random.default_rng(self.seed)
            rng.shuffle(case_ids)

            n_cases = len(case_ids)
            n_train = int(n_cases * self.train_ratio)
            n_val   = int(n_cases * self.val_ratio)

            train_cases = case_ids[:n_train]
            val_cases   = case_ids[n_train:n_train + n_val]
            test_cases  = case_ids[n_train + n_val:]

            splits[cid] = {
                "train": [s for c in train_cases for s in cases[c]],
                "val":   [s for c in val_cases   for s in cases[c]],
                "test":  [s for c in test_cases  for s in cases[c]],
                "total": len(samples),
            }
        return splits

    def build_global_test_set(self, splits: Dict) -> List[dict]:
        return [s for data in splits.values() for s in data["test"]]

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "num_clients": len(self.clients),
            "total_samples": sum(len(v) for v in self.clients.values()),
            "clients": {
                cid: {"n": len(s), "project": s[0]["project"]}
                for cid, s in self.clients.items()
            },
        }

    def filter_by_project(self, project: str) -> "TCGASurvivalPartitioner":
        """Return a copy containing only clients from one cancer type (e.g. TCGA-BRCA)."""
        filtered = TCGASurvivalPartitioner(
            feature_dir=str(self.feature_dir.parent),
            model_name=self.model_name,
            clinical_csv=self.clinical_csv,
            min_samples=self.min_samples,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        filtered.clients = {
            cid: samples for cid, samples in self.clients.items()
            if samples[0]["project"] == project
        }
        return filtered


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("/home/holiday01/fl_wsi/configs/fl_config.yaml"))
    partitioner = TCGAFLPartitioner(
        feature_dir=cfg["data"]["feature_dir"],
        model_name="UNI_v2",
        metadata_tsv=cfg["data"]["metadata_tsv"],
        min_samples=cfg["federated"]["min_samples_per_client"],
        seed=cfg["experiment"]["seed"],
    )
    partitioner.scan_features()
    stats = partitioner.summary()
    print(f"\nTotal clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    top = sorted(stats["clients"].items(), key=lambda x: -x[1]["n"])[:10]
    for cid, info in top:
        print(f"  {cid}: {info['n']} samples")

    out = "/home/holiday01/fl_wsi/data/partition_UNI_v2.json"
    partitioner.save_partition(out)
