#!/usr/bin/env python
"""
Cohort / federation statistics.
  * per-cancer: clients, slides, patients, median slides per client, slides per patient
  * unique TSS codes overall and whether a TSS code appears in > 1 project
  * institution-level partition (TSS -> contributing institution via the GDC TSS table)
  * slide type composition (DX / TS / BS / MS)
Outputs analysis/tables/T_cohort.csv, T_institutions.csv, cohort_stats.json
"""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "code"))
from fedfm.data import tss_institution_map, CLASS_NAMES

OUT = REV / "analysis" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(REV / "data" / "cohort_classification.json"))

rows = []
for lab, name in enumerate(CLASS_NAMES):
    ss = [s for s in S if s["label"] == lab]
    clients = Counter(s["client_id"] for s in ss)
    pats = set(s["case_id"] for s in ss)
    rows.append(dict(cancer=name, clients=len(clients), slides=len(ss), patients=len(pats),
                     median_slides_per_client=float(np.median(list(clients.values()))),
                     min_slides=min(clients.values()), max_slides=max(clients.values()),
                     slides_per_patient=len(ss) / len(pats),
                     train=sum(s["split"] == "train" for s in ss), val=sum(s["split"] == "val" for s in ss),
                     test=sum(s["split"] == "test" for s in ss)))
T = pd.DataFrame(rows)
tot = dict(cancer="Total", clients=T.clients.sum(), slides=T.slides.sum(), patients=len(set(s["case_id"] for s in S)),
           median_slides_per_client=float(np.median(list(Counter(s["client_id"] for s in S).values()))),
           min_slides=T.min_slides.min(), max_slides=T.max_slides.max(),
           slides_per_patient=len(S) / len(set(s["case_id"] for s in S)),
           train=T.train.sum(), val=T.val.sum(), test=T.test.sum())
T = pd.concat([T, pd.DataFrame([tot])]); T.to_csv(OUT / "T_cohort.csv", index=False)
print(T.to_string())

# slides per patient
per_pat = Counter(s["case_id"] for s in S)
spp = np.array(list(per_pat.values()))
# TSS codes
tss_proj = defaultdict(set)
for s in S:
    tss_proj[s["tss"]].add(s["project"])
multi = {t: p for t, p in tss_proj.items() if len(p) > 1}
# institutions
m = tss_institution_map()
inst_clients = defaultdict(set); inst_slides = Counter(); inst_cancers = defaultdict(set)
unmapped = set()
for s in S:
    inst = m.get(s["tss"])
    if inst is None:
        unmapped.add(s["tss"]); inst = s["tss"]
    inst_clients[inst].add(s["client_id"]); inst_slides[inst] += 1; inst_cancers[inst].add(s["project"])
I = pd.DataFrame([dict(institution=i, tss_clients=len(inst_clients[i]), slides=inst_slides[i],
                       cancer_types=len(inst_cancers[i]), cancers=";".join(sorted(c.replace("TCGA-", "") for c in inst_cancers[i])))
                  for i in inst_clients]).sort_values("slides", ascending=False)
I.to_csv(OUT / "T_institutions.csv", index=False)
stats = dict(n_slides=len(S), n_patients=len(per_pat), slides_per_patient_mean=float(spp.mean()),
             slides_per_patient_median=float(np.median(spp)), slides_per_patient_max=int(spp.max()),
             patients_multi_slide=int((spp > 1).sum()),
             slide_types=Counter(s["slide_type"] for s in S),
             n_unique_tss=len(tss_proj), tss_in_multiple_projects=len(multi),
             n_institutions=len(inst_clients), institutions_multi_cancer=int((I.cancer_types > 1).sum()),
             institutions_top=I.head(10).to_dict("records"), unmapped_tss=sorted(unmapped),
             client_size_train=dict(Counter(s["client_id"] for s in S if s["split"] == "train")))
json.dump(stats, open(REV / "analysis" / "cohort_stats.json", "w"), indent=1, default=str)
print({k: v for k, v in stats.items() if k not in ("institutions_top", "client_size_train")})
print(I.head(20).to_string())
