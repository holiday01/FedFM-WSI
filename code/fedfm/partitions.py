"""
Client partitions.  The patient-level train/val/test split is NEVER changed:
only the assignment of training (and validation) patients to clients varies, so every
partition is evaluated on the identical held-out test set.

  project_tss   : legacy federation (107 Project_TSS clients, one cancer type each)
  institution   : TSS codes mapped to contributing institution (GDC TSS code table);
                  all codes of one institution form one client, which may hold several
                  cancer types
  labelskew_<a> : 107 clients with exactly the Project_TSS client sizes (quantity skew
                  preserved), class composition drawn from Dirichlet(a); a -> inf is IID
  iid           : same sizes, uniformly random patients
"""
from collections import defaultdict, Counter

import numpy as np

from .data import tss_institution_map


def client_assignment(samples, partition, seed=2026):
    if partition == "project_tss":
        return [s["client_id"] for s in samples]
    if partition in ("institution", "institution_all"):
        # "institution": test slides keep their Project_TSS id (per-client reporting unchanged);
        # "institution_all": test slides are also grouped by institution, which is required for
        # FedBN's per-client evaluation with institution-level BN states.
        m = tss_institution_map()
        keep_test = partition == "institution"
        return [("INST:" + m.get(s["tss"], s["tss"])) if (s["split"] != "test" or not keep_test) else s["client_id"]
                for s in samples]
    if partition.startswith("labelskew_") or partition == "iid":
        alpha = None if partition == "iid" else float(partition.split("_")[1])
        return _label_skew(samples, alpha, seed)
    raise ValueError(partition)


def _label_skew(samples, alpha, seed):
    rng = np.random.default_rng(seed)
    out = [s["client_id"] for s in samples]            # test samples keep their original id
    for split in ("train", "val"):
        pat_label, pat_members = {}, defaultdict(list)
        quota = Counter()
        seen = set()
        for i, s in enumerate(samples):
            if s["split"] != split:
                continue
            pat_members[s["case_id"]].append(i)
            pat_label[s["case_id"]] = s["label"]
            if s["case_id"] not in seen:
                quota[s["client_id"]] += 1               # patient quota per original client
                seen.add(s["case_id"])
        clients = sorted(quota)
        K = len(clients)
        rem = np.array([quota[c] for c in clients], dtype=float)
        if alpha is None:
            pref = np.ones((K, 9))
        else:
            pref = rng.dirichlet(np.full(9, alpha), size=K)
        pats = list(pat_members)
        rng.shuffle(pats)
        for p in pats:
            c = pat_label[p]
            w = pref[:, c] * (rem > 0)
            if w.sum() <= 0:
                w = (rem > 0).astype(float)
            k = rng.choice(K, p=w / w.sum())
            rem[k] -= 1
            for i in pat_members[p]:
                out[i] = f"SKEW:{clients[k]}"
    return out


def partition_summary(samples, client_of):
    by = defaultdict(Counter)
    for s, c in zip(samples, client_of):
        if s["split"] == "train":
            by[c][s["label"]] += 1
    ncls = [len(v) for v in by.values()]
    maxfrac = [max(v.values()) / sum(v.values()) for v in by.values()]
    return dict(n_clients=len(by), mean_classes_per_client=float(np.mean(ncls)),
                clients_multi_class=int(sum(n > 1 for n in ncls)),
                mean_majority_fraction=float(np.mean(maxfrac)))
