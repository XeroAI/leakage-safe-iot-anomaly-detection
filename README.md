# Leakage-Safe Event-Level Evaluation of Compact Detectors for Multivariate IoT Anomaly Detection

This repository contains the **code required to reproduce** the experiments in the paper.

**This paper is submitted to *Internet of Things* — Elsevier.**

The contribution is a leakage-safe event-level evaluation protocol and an efficiency–accuracy comparison of compact CNN / CNN–Transformer detectors on MSL and SMD. It is **not** a state-of-the-art F1 claim and **not** a deployment recommendation.

---

## 1. Paper

- **Title:** Leakage-Safe Event-Level Evaluation of Compact Detectors for Multivariate IoT Anomaly Detection
- **Venue:** *Internet of Things* — Elsevier (submitted)
- **Authors:** Adil Afzal (corresponding author), Saleh Alghamdi, Sultan Alahmari, Sultan Almutairi, Muhammad Rizwan, Ovidiu Bagdasar, Natalia Kryvinska

## 2. What this repository includes

| File | Role |
|------|------|
| `model.py` | LST-TFDN, CNN-only, Standard Transformer, Faithful Encoding, linearized attention |
| `data_loader.py` | Event-level holdout, leaky stratified split, MinMaxScaler (no ε), windowing |
| `train.py` | Single-run training and validation-F1 threshold search |
| `phase1_msl_unified.py` | Primary MSL 5-seed campaign (baselines, AUROC/AUPRC, FLOPs, class-weight sweep) |
| `baselines_event.py` | Like-for-like MSL baselines under event holdout |
| `ablation_study.py` | MSL ablations (CNN-only, standard attention, SPE, no CNN) |
| `tierB_leaky_vs_event.py` | Matched leaky stratified vs event-level comparison |
| `revision_experiments.py` | Protocol checks, shallow baselines (logistic regression, HistGB), figure overlays |
| `smd_event_benchmark.py` | Per-machine SMD event-level benchmark |
| `benchmark_smd_all.py` | SMD machine listing helper |
| `preprocess_smd.py` / `preprocess_smd_all.py` | Convert SMD `.txt` files to `.npy` |
| `explainability.py` | Illustrative saliency maps (requires a checkpoint; see §7) |
| `multi_seed_eval.py` | Repeated-seed wrapper around `train.py` |

**Not included:** model weights and raw datasets.

## 3. Requirements

Python 3.10+ is recommended. Install:

```bash
pip install -r requirements.txt
```

CPU-only PyTorch is sufficient. Dedicated GPU is optional.

## 4. Datasets

This repository does **not** ship MSL or SMD arrays. Place the public files as follows after download.

## MSL (Mars Science Laboratory)

Source: NASA/JPL Telemanom release  
https://github.com/khundman/telemanom  
Raw archive: https://s3-us-west-2.amazonaws.com/telemanom/data.zip

Expected layout:

```
datasets/MSL/MSL_train.npy
datasets/MSL/MSL_test.npy
datasets/MSL/MSL_test_label.npy
```

## SMD (Server Machine Dataset)

Source: OmniAnomaly  
https://github.com/NetManAIOps/OmniAnomaly  
Dataset folder: `ServerMachineDataset`

Expected layout:

```
datasets/SMD/train/machine-*.txt
datasets/SMD/test/machine-*.txt
datasets/SMD/test_label/machine-*.txt
```

Convert one machine (or all machines) to `.npy` with:

```bash
python preprocess_smd.py
python preprocess_smd_all.py
```

## 5. Protocol (paper default)

- Window length \(T=100\), stride \(\Delta=10\)
- Event-level holdout, event-split seed \(s_{\mathrm{evt}}=42\)
- Initialization seeds `{42, 123, 456, 789, 1024}` on MSL
- Threshold \(\tau^*\) chosen on validation F1 over `{0.01, …, 0.98}`
- Compressed CNN length \(T'=50\)

Headline MSL numbers in the paper use this protocol. A matched **leaky stratified** split is provided only for the protocol comparison.

## 6. Reproducing the experiments

From this folder, after datasets are in place:

```bash
# Protocol checks + shallow baselines (logistic regression, HistGB)
python revision_experiments.py --phase verify
python revision_experiments.py --phase shallow

# Primary MSL 5-seed campaign
python phase1_msl_unified.py

# Ablations (same event split)
python ablation_study.py

# Leaky vs event-level comparison (3 seeds)
python tierB_leaky_vs_event.py

# SMD per-machine event-level benchmark (27 machines × 3 seeds)
python smd_event_benchmark.py

# Optional: single LST-TFDN run (MSL by default; set DATASET_NAME in train.py for SMD)
python train.py
```


## 7. Model weights

**Trained model weights are not included in this repository.**

Weights will be available on request to the corresponding author on a reasonable request.

## 8. License and data

- **Code** in this folder is released to support reproduction of the submitted paper.
- **MSL** and **SMD** remain under the licenses of their original public releases (Telemanom / OmniAnomaly). Cite those sources when using the data.

## 9. Citation

Available soon (Submitted to *Internet of Things* — Elsevier).
