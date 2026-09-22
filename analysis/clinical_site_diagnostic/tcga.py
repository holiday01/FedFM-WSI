"""Loaders for the local TCGA validation cohort (BRCA / COAD / STAD).

Everything here is already on disk:
  * patch features  trident_processed/20x_224px_0px_overlap/features_phikon_v2
  * slide-level FM baselines  /mnt/10t/cached_slide_features_npy/<model>
  * bulk RNA-seq    drug_tcga/TCGA_Foundation_Data/<proj>/GDC_data  (STAR counts)
  * clinical / OS   foundation/TCGA_clinical_BRCA_COAD_STAD.csv
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

RNA_ROOT = "/home/holiday01/drug_tcga/TCGA_Foundation_Data"
CLINICAL = "/home/holiday01/foundation/TCGA_clinical_BRCA_COAD_STAD.csv"
BASELINE_ROOT = "/mnt/10t/cached_slide_features_npy"
WSI_ROOT = "/mnt/10t/holiday/1019_image"
CANCERS = ["TCGA-BRCA", "TCGA-COAD", "TCGA-STAD"]


def patient_of(slide_id: str) -> str:
    """TCGA-BH-A0DP-01A-02-BSB.<uuid>  ->  TCGA-BH-A0DP"""
    return slide_id[:12]


def is_primary_tumor(slide_id: str) -> bool:
    """Sample-type code 01-09 is tumour; 10-19 is normal. Keep tumour only."""
    parts = slide_id.split("-")
    if len(parts) < 4 or len(parts[3]) < 2 or not parts[3][:2].isdigit():
        return False
    return 1 <= int(parts[3][:2]) <= 9


def slide_to_cancer() -> dict[str, str]:
    m = {}
    for d in glob.glob(os.path.join(WSI_ROOT, "TCGA-*")):
        c = os.path.basename(d)
        for s in glob.glob(os.path.join(d, "**", "*.svs"), recursive=True):
            m[os.path.basename(s)[:-4]] = c
    return m


def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(CLINICAL)
    df = df[df.Project.isin(CANCERS)].copy()
    df = df.dropna(subset=["OS_time", "OS_status"])
    df = df[df.OS_time > 0]
    return df.drop_duplicates("submitter_id").set_index("submitter_id")


def load_rna(cancers: list[str] | None = None, value: str = "tpm_unstranded",
             cache: str | None = "/home/holiday01/visium_wsi/data/tcga_rna.parquet") -> pd.DataFrame:
    """Bulk RNA-seq as patients x gene_symbol (TPM).

    GDC STAR files are keyed by file name; `<proj>_file_log.csv` maps file name to the
    aliquot barcode, whose first 12 characters are the patient. Parsing ~300 60k-row TSVs
    takes minutes, so the assembled matrix is cached.
    """
    cancers = cancers or CANCERS
    if cache and cancers == CANCERS and os.path.exists(cache):
        return pd.read_parquet(cache)

    cols, index = [], []
    for proj in cancers:
        log_p = os.path.join(RNA_ROOT, proj, f"{proj}_file_log.csv")
        if not os.path.exists(log_p):
            continue
        log = pd.read_csv(log_p)
        fn2case = dict(zip(log.file_name, log.cases))
        for f in glob.glob(os.path.join(RNA_ROOT, proj, "GDC_data", "**", "*.tsv"), recursive=True):
            case = fn2case.get(os.path.basename(f))
            if case is None or not is_primary_tumor(case):
                continue
            t = pd.read_csv(f, sep="\t", skiprows=1, low_memory=False)
            t = t[t.gene_id.astype(str).str.startswith("ENSG")]
            t = t[t.gene_type == "protein_coding"]
            s = t.groupby("gene_name")[value].max()
            cols.append(s)
            index.append(patient_of(case))

    if not cols:
        return pd.DataFrame()
    X = pd.concat(cols, axis=1).T
    X.index = index
    # A few patients have >1 aliquot sequenced; average them.
    X = X.groupby(level=0).mean()
    if cache and cancers == CANCERS:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        X.to_parquet(cache)
    return X


def load_baseline_slide_features(model: str) -> dict[str, np.ndarray]:
    out = {}
    for p in glob.glob(os.path.join(BASELINE_ROOT, model, "*.npy")):
        out[os.path.basename(p)[:-4]] = np.load(p).astype(np.float32)
    return out


def baseline_patient_matrix(model: str, cache_dir: str = "/home/holiday01/visium_wsi/data/baseline_cache"
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level baseline features, cached — each model is ~5.7k separate .npy reads."""
    cache = os.path.join(cache_dir, f"{model}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["patients"], z["X"]
    pats, X = slides_to_patients(load_baseline_slide_features(model), tumor_only=True)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, patients=pats, X=X)
    return pats, X


def slides_to_patients(
    feats: dict[str, np.ndarray], tumor_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Average a slide-level feature dict up to one vector per patient."""
    bag: dict[str, list[np.ndarray]] = {}
    for sid, v in feats.items():
        if tumor_only and not is_primary_tumor(sid):
            continue
        bag.setdefault(patient_of(sid), []).append(v)
    pats = sorted(bag)
    return np.array(pats), np.stack([np.mean(bag[p], axis=0) for p in pats])
