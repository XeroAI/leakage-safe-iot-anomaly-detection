# Leakage-safe evaluation of compact detectors on multivariate IoT telemetry

Final training and evaluation code for the *Scientific Reports* manuscript.

This package reproduces **event-exclusive** supervised window scoring on MSL/SMD (and cross-stream checks), with a locked shallow baseline `HistGB_5S_fixedHP`. It is an evaluation-reliability study, not a SOTA F1 contest.

---

## Contents

| File | Role |
|------|------|
| `model.py` | LST-TFDN, CNN-only, Standard Transformer |
| `data_loader.py` | Event-exclusive and stratified splits, windowing |
| `train.py` | Single-run training + validation F1 threshold search |
| `msl_fixed_partition.py` | Fixed longest-first MSL 5-seed campaign |
| `baselines_event.py` | Like-for-like fixed-partition baselines |
| `ablation_study.py` | Module ablations on the fixed partition |
| `leaky_vs_event.py` | Stratified (leaky) vs event-exclusive comparison |
| `protocol_checks.py` | Protocol checks and shallow baselines |
| `buffer_b_sweep.py` | Isolation Forest / PCA and context-buffer \(B\) sweep |
| `msl_event_splits.py` | Freeze 20 nested event partitions |
| `msl_multipartition_train.py` | 20-partition campaign (CNN / LST / StdT / HistGB) |
| `msl_event_cluster_bootstrap.py` | Event-cluster bootstrap CIs |
| `protocol_2x2.py` | Population-pool \(2\times2\) protocol contrast |
| `cross_stream_prepare.py` / `cross_stream_train.py` | PSM / SMAP / MSL-entity scope |
| `smd_event_benchmark.py` | Per-machine SMD event holdout |
| `preprocess_smd.py` / `preprocess_smd_all.py` | SMD text → npy |
| `explainability.py` | Illustrative saliency |
| `multi_seed_eval.py` | Multi-seed wrapper around `train.py` |

**Not included:** weights, raw datasets, manuscript LaTeX, exploratory campaigns.

## Setup

```bash
pip install -r requirements.txt
```

Place public MSL/SMD arrays under `datasets/` (see `datasets/README.md`).

## Reproduce main results

```bash
# Fixed-partition tables
python protocol_checks.py --phase verify
python protocol_checks.py --phase shallow
python buffer_b_sweep.py
python msl_fixed_partition.py
python ablation_study.py
python leaky_vs_event.py

# Multi-partition campaign (primary ranking tables)
python msl_multipartition_train.py --freeze-only
python msl_multipartition_train.py --phase all
python msl_event_cluster_bootstrap.py

# Protocol 2x2 (assignment sensitivity)
python protocol_2x2.py --seeds 42 123 456 789 1024 --inits 42 123 456

# Cross-stream + SMD
python cross_stream_prepare.py
python cross_stream_train.py
python smd_event_benchmark.py
```

Outputs are written under `msl_multipartition/`, `protocol_2x2/`, and `cross_stream/` (gitignored).

## Protocol defaults

- \(T=100\), \(\Delta=10\), \(B=200\)
- Locked HistGB: `max_depth=6`, `lr=0.05`, `max_iter=300`, `random_state=42`
- Threshold grid `{0.01,…,0.98}` on validation F1

## License / data

Code supports manuscript reproduction. MSL/SMD remain under Telemanom / OmniAnomaly licenses.
