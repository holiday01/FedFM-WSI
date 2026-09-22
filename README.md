# FedFM-WSI

Benchmarking pathology foundation models (FMs) as **frozen feature extractors** in
**federated learning** for pan-cancer whole-slide-image (WSI) analysis.

Seven encoders (UNI v2, Virchow2, Phikon v2, CONCH v1.5, CTransPath, Midnight-12k,
ResNet50) are compared on two tasks over TCGA slide-level features:

* nine-class cancer-type classification in a 107-client federation
  (one client per TCGA project x tissue source site), and
* within-cancer survival (BRCA, COAD, STAD) with a federated Cox model.

Federated algorithms: FedAvg (class-balanced aggregation), FedProx, SCAFFOLD (Option II),
FedBN (local batch-norm statistics) and an aggregated-BN reference (FedAvgBN);
client sampling: stratified-by-cancer, uniform, UCB1 and PathologyAware selectors.
Every federated result is paired with centralized training on the pooled training
slides under the same optimiser, and every comparison is repeated over seeds with
validation-only model selection. A CPTAC (LUAD, PDA) cohort is used for external
evaluation of the trained heads.

## Layout

| Folder | Content |
|---|---|
| `code/fedfm/` | engine: data layer, client partitions, FL loop, centralized baselines, survival, heads and losses, metrics, selectors |
| `code/legacy_copy/` | earlier version of the pipeline; the Project_TSS partitioner and the UCB / PathologyAware selectors are imported from here unchanged |
| `scripts/` | experiment runners (`run_cls.py`, `run_surv.py`, `run_stream_*.sh`), CPTAC feature builder, external evaluation |
| `data/` | cohort definitions (JSON, one record per slide), GDC tissue-source-site table, CPTAC manifest |
| `results/cls/<grid>/<encoder>/` | one JSON per run (key = hash of the configuration); `results/surv/`, `results/external/` |
| `analysis/` | aggregation (`aggregate_cls.py`, `aggregate_surv.py`, `aggregate_external.py`), cohort statistics, cost table, figures; `checkpoint_precision.py` (float16-checkpoint diagnostic); `clinical_site_diagnostic/` (archived PCA-Cox clinical-baseline / site-robustness diagnostic with its output); `tables/*.csv` are the aggregated outputs |
| `tests/` | unit tests and invariants of the engine (`python tests/check_engine.py`) |

The run JSONs, the prediction files (`*_pred.npz`) and the aggregated tables are included,
so `analysis/aggregate_cls.py`, `analysis/aggregate_surv.py` and `analysis/aggregate_external.py` run from the
repository as is. Every classification metric in `analysis/tables/` is computed from the archived per-slide
predictions: runs archived before 2026-09-22 hold float16 class probabilities only (class = argmax; their training-time
JSON metrics can differ for near-tied slides, see `tables/T_prediction_source_audit.csv`), later runs also hold the
integer class (`pred`) and float32 probabilities, and checkpoints written from then on are float32.
Checkpoints (`*.pt`, 11 GB) are not tracked; `scripts/eval_external.py` needs them and can
only be re-run after the `main`, `sgd` and `central` grids have been regenerated with `--save-ckpt`.

## Slide-level features

A slide is represented by eight per-dimension statistics of its tile embeddings
(mean, standard deviation, max, min, 25th, 50th, 75th and 90th percentile),
concatenated to an `8 x D` vector. `scripts/build_cptac_features.py` applies the
same aggregation to CPTAC tile features. The TCGA slide-feature cache is expected at
`FEATURE_ROOT` in `code/fedfm/data.py` (one `.npy` per slide and encoder) and the
clinical table at `CLINICAL_CSV`; adjust both paths for your installation.

## Protocol v2 (what the engine enforces)

* one common cohort for every encoder (slides whose features exist for all seven encoders),
  patient-level 70/10/20 split inside each client, identical test set for every partition;
* the last incomplete local mini-batch is **not** dropped (the legacy `drop_last=True`
  rule silently gave zero local steps to small clients; `legacy_drop_last=True` reproduces it);
* checkpoint selection and early stopping use the pooled validation split only;
  learning rate, mu and participation are selected on validation and reported on test;
* explicit seeds for initialisation, client sampling, shuffling and dropout;
* survival: Breslow partial likelihood with tied-time risk sets at the patient level,
  clients with fewer than five patients excluded, C-index with the lifelines tie convention,
  site-stratified centralized Cox as reference; legacy modes reproduce the earlier loss;
* uncertainty: seed s.d. for every cell, two-level (seed + patient cluster) bootstrap for
  paired differences, paired-seed t-intervals.

## Reproduce

```bash
export PYTHONDONTWRITEBYTECODE=1
python tests/check_engine.py                                   # unit tests and invariants
bash scripts/run_stream_A.sh                                   # main protocol (Adam), centralized, mu / lr / participation sweeps, linear probes, controls, FedBN on institutions
bash scripts/run_stream_B.sh                                   # SGD optimiser control, partitions, selection policies, SGD mu sweep
bash scripts/run_stream_C.sh                                   # within-cancer survival
bash scripts/run_stream_D.sh                                   # legacy-batching reproduction (waits for A)
bash scripts/run_stream_G.sh; bash scripts/run_stream_H.sh     # standardised-feature follow-ups, FedBN / SGD on institutions, linear probes
bash scripts/run_stream_v2a.sh; bash scripts/run_stream_v2b.sh; bash scripts/run_stream_v2c.sh
                                                               # centralized Adam / SGD, selection (uniform arm), FedProx on all partitions, FedAvgBN, Adam lr per algorithm
for c in BRCA COAD STAD; do python scripts/run_surv.py --grid all --cancers $c --seeds 0 1 2 3 4; done
python scripts/build_cptac_features.py && python scripts/eval_external.py
python analysis/cohort_stats.py && python analysis/aggregate_cls.py && python analysis/aggregate_surv.py
python analysis/aggregate_external.py && python analysis/checkpoint_precision.py   # CPTAC paired bootstrap; checkpoint diagnostic (needs *.pt)
python analysis/cost_table.py && python analysis/make_figures.py   # figures/*.pdf
```

All runners are resumable: a run whose configuration hash already has a JSON is skipped.
Single grids: `python scripts/run_cls.py --grid main --fms UNI_v2 Conch_v15 --seeds 0 1 2 3 4`
(grids are listed in `scripts/run_cls.py`).

## Environment

Python 3.11, torch 2.10 (CUDA 12.8), scikit-learn 1.9, lifelines 0.30, numpy 2.1, pandas 2.3,
scipy 1.17, matplotlib 3.10; one 16 GB GPU. See `requirements.txt`.

## Data

TCGA whole-slide images: Genomic Data Commons (https://portal.gdc.cancer.gov).
CPTAC LUAD and PDA: https://www.cancerimagingarchive.net.
Tile embeddings were extracted with each encoder at 20x on 256 x 256 patches; only the
slide-level aggregated features are used here.

## License

MIT (see `LICENSE`).
