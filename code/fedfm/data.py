"""
Data layer (Protocol v2).

Differences from the legacy pipeline in code/legacy_copy:
  * A single COMMON cohort is used for every foundation model: only slides whose
    cached slide-level feature file exists for all seven encoders are retained
    (5,208 slides).  The legacy pipeline re-partitioned each encoder's own file
    list, so train/val/test differed slightly between encoders (test-case Jaccard
    0.97), which precludes paired comparisons.
  * The partition algorithm itself is unchanged and is imported verbatim from the
    legacy code (legacy_copy/data/partition.py): Project_TSS clients, >=10 slides,
    patient(case)-level 70/10/20 split inside each client with seed 42.
  * Features are loaded once into a float32 matrix (cached as .npy in
    data/cache) instead of being re-read from disk every round.
    This does not change any value.
"""
import os
import io
import re
import sys
import json
import contextlib
from pathlib import Path
from collections import defaultdict

import numpy as np

REV_ROOT = Path(__file__).resolve().parents[2]            # repository root
LEGACY = REV_ROOT / "code" / "legacy_copy"
FEATURE_ROOT = Path("/mnt/10t/cached_slide_features_npy")
CACHE_DIR = REV_ROOT / "data" / "cache"
CLINICAL_CSV = "/home/holiday01/foundation/TCGA_clinical_BRCA_COAD_STAD.csv"

FMS = ["UNI_v2", "Virchow2", "Phikon_v2", "Conch_v15", "CTransPath", "Midnight12k", "ResNet50"]
FM_LABEL = {"UNI_v2": "UNI v2", "Virchow2": "Virchow2", "Phikon_v2": "Phikon v2",
            "Conch_v15": "CONCH v1.5", "CTransPath": "CTransPath",
            "Midnight12k": "Midnight-12k", "ResNet50": "ResNet50"}
CLASS_NAMES = ["BRCA", "COAD", "STAD", "LGG", "LUAD", "HNSC", "SKCM", "CESC", "PAAD"]

sys.path.insert(0, str(LEGACY))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from data.partition import (TCGAFLPartitioner, TCGASurvivalPartitioner,  # noqa: E402
                            build_case_to_project_map, CANCER_TYPE_MAP)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def common_filenames():
    """Slide files present for all seven encoders."""
    sets = [set(os.listdir(FEATURE_ROOT / m)) for m in FMS]
    return set.intersection(*sets)


def slide_type(fname):
    m = re.match(r"TCGA-\w\w-\w{4}-\w{3}-\w{2}-([A-Z]{2})", fname)
    return m.group(1) if m else "NA"


def build_classification_cohort(seed=42, min_samples=10):
    """Return a dict describing the common classification cohort and its split.

    samples: list of dicts (filename, case_id, project, tss, client_id, label, split)
    """
    keep = common_filenames()
    p = TCGAFLPartitioner(str(FEATURE_ROOT), "UNI_v2", min_samples=min_samples, seed=seed)
    _quiet(p.scan_features)
    # scan_features already applied min_samples on the UNI_v2 file list; re-filter on the
    # common file set and re-apply the threshold so the rule is identical.
    p.clients = {cid: [s for s in v if s["filename"] in keep] for cid, v in p.clients.items()}
    p.clients = {cid: v for cid, v in p.clients.items() if len(v) >= min_samples}
    splits = p.split_clients()
    samples = []
    for cid, d in splits.items():
        for sp in ("train", "val", "test"):
            for s in d[sp]:
                samples.append(dict(filename=s["filename"], case_id=s["case_id"],
                                    project=s["project"], tss=s["tss"], client_id=cid,
                                    label=int(s["cancer_label"]), split=sp,
                                    slide_type=slide_type(s["filename"])))
    samples.sort(key=lambda s: s["filename"])
    return samples


def load_feature_matrix(fm, filenames):
    """float32 matrix aligned with `filenames` (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"{fm}_{len(filenames)}.npy"
    idx_key = CACHE_DIR / f"{fm}_{len(filenames)}.json"
    if key.exists() and idx_key.exists():
        if json.load(open(idx_key)) == list(filenames):
            return np.load(key, mmap_mode="r")
    X = None
    for i, f in enumerate(filenames):
        a = np.load(FEATURE_ROOT / fm / f)
        if a.ndim > 1:
            a = a.mean(axis=0)
        if X is None:
            X = np.zeros((len(filenames), a.shape[0]), dtype=np.float32)
        X[i] = a.astype(np.float32)
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        print(f"[data] WARNING {fm}: {bad.sum()} slides with non-finite features set to 0")
        X[bad] = 0.0
    np.save(key, X)
    json.dump(list(filenames), open(idx_key, "w"))
    return np.load(key, mmap_mode="r")


# ─────────────────────────── survival cohort ───────────────────────────

def build_survival_cohort(project, seed=42, min_samples=5):
    """Within-cancer survival cohort (common file set), split as in the legacy code.

    The legacy SurvivalPartitioner iterates an *unsorted* glob; to make the split
    reproducible we sort the file list (documented deviation).
    """
    import pandas as pd
    keep = common_filenames()
    surv = pd.read_csv(CLINICAL_CSV)
    surv = surv.dropna(subset=["OS_time", "OS_status"])
    surv = surv[surv["OS_time"] > 0]
    smap = {r.submitter_id: r for r in surv.itertuples()}
    case_to_proj = _quiet(build_case_to_project_map)
    clients = defaultdict(list)
    for f in sorted(keep):
        name = f.split(".")[0]
        parts = name.split("-")
        case = "-".join(parts[:3])
        if case_to_proj.get(case) != project or case not in smap:
            continue
        r = smap[case]
        clients[f"{project}_{parts[1]}"].append(dict(
            filename=f, case_id=case, project=project, tss=parts[1],
            client_id=f"{project}_{parts[1]}", os_time=float(r.OS_time),
            os_status=int(r.OS_status), age=r.age_at_index,
            stage=r.ajcc_pathologic_stage, slide_type=slide_type(f)))
    # inclusion rule: at least `min_samples` matched PATIENTS per client (the legacy code
    # counted slides)
    clients = {k: v for k, v in clients.items() if len(set(s["case_id"] for s in v)) >= min_samples}
    samples = []
    for cid, ss in clients.items():
        cases = defaultdict(list)
        for s in ss:
            cases[s["case_id"]].append(s)
        case_ids = list(cases.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(case_ids)
        n = len(case_ids)
        ntr, nva = int(n * 0.7), int(n * 0.1)
        for j, c in enumerate(case_ids):
            sp = "train" if j < ntr else ("val" if j < ntr + nva else "test")
            for s in cases[c]:
                samples.append(dict(s, split=sp))
    samples.sort(key=lambda s: s["filename"])
    return samples


# ─────────────────────────── institution map ───────────────────────────

_INSTITUTION_ALIASES = {
    "MD Anderson Cancer Center": "MD Anderson",
    "Mayo Clinic - Rochester": "Mayo",
    "Case Western - St Joes": "Case Western",
}


def tss_institution_map():
    import pandas as pd
    t = pd.read_csv(REV_ROOT / "data" / "tcga_tss_codes.csv", dtype=str)
    m = {}
    for r in t.itertuples():
        name = str(r.site).strip()
        m[str(r.tss).zfill(2)] = _INSTITUTION_ALIASES.get(name, name)
    return m
