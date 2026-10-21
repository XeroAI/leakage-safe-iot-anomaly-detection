"""
Same-protocol unsupervised references + context-buffer (B) sweep on MSL.

Fits Isolation Forest and PCA reconstruction error on the same 5S window
statistics used by the shallow supervised baselines, under the published
longest-first event split (event_seed=42, seeded_variation=False).

Also sweeps B in {50, 100, 200, 400} and reports:
  - split sizes / test base rate / always-positive F1
  - Isolation Forest, PCA, and HistGB (matched paper HPs) under each B

Outputs: unsupervised_b_sweep_results.json (+ concise TXT summary)

CPU-only. Resume-safe: re-running overwrites the JSON with a full recompute
(cheap; seconds to a few minutes).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import TimeSeriesDataset, get_dataloaders
from phase1_msl_unified import BATCH_SIZE, STRIDE, WINDOW_SIZE

RESULTS_JSON = "unsupervised_b_sweep_results.json"
RESULTS_TXT = "unsupervised_b_sweep_results.txt"

# Matched HistGB HPs from v7_fill_metrics.json / paper Table trivial
HISTGB_PARAMS = dict(
    learning_rate=0.03,
    max_leaf_nodes=15,
    l2_regularization=0.0,
    max_iter=100,
    random_state=42,
)

DEFAULT_B_VALUES = [50, 100, 200, 400]
PRIMARY_B = 200


def safe_auroc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _window_feature_matrix(
    dataset: TimeSeriesDataset, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sensor mean, std, min, max, slope over each window (5S)."""
    xs, ys = [], []
    for wi in indices:
        x, y = dataset[int(wi)]  # x: (S, T)
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        amin = x.min(axis=1)
        amax = x.max(axis=1)
        t = np.arange(x.shape[1], dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-8)
        slope = (x * t[None, :]).mean(axis=1)
        feat = np.concatenate([mean, std, amin, amax, slope], axis=0)
        xs.append(feat)
        ys.append(float(y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def _best_threshold(y: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.01, 0.99, 0.01):
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t, best_f


def _metrics_from_scores(
    name: str,
    yte: np.ndarray,
    scores_te: np.ndarray,
    thresh: float,
    trivial_f1: float,
    pi: float,
    extra: Optional[Dict] = None,
) -> Dict:
    pred = (scores_te >= thresh).astype(int)
    oracle_t, oracle_f = _best_threshold(yte, scores_te)
    row = {
        "name": name,
        "test_base_rate": float(pi),
        "trivial_f1": float(trivial_f1),
        "best_thresh": float(thresh),
        "test_f1": float(f1_score(yte, pred, zero_division=0)),
        "test_precision": float(precision_score(yte, pred, zero_division=0)),
        "test_recall": float(recall_score(yte, pred, zero_division=0)),
        "test_mcc": float(matthews_corrcoef(yte, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "test_balanced_acc": float(balanced_accuracy_score(yte, pred)),
        "test_auroc": safe_auroc(yte, scores_te),
        "test_auprc": safe_auprc(yte, scores_te),
        "oracle_test_f1": float(oracle_f),
        "oracle_thresh": float(oracle_t),
    }
    row["delta_f1_vs_trivial"] = row["test_f1"] - trivial_f1
    row["oracle_delta_f1_vs_trivial"] = oracle_f - trivial_f1
    if extra:
        row.update(extra)
    return row


def _minmax01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _load_split(B: int):
    train_loader, val_loader, test_loader = get_dataloaders(
        "MSL",
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split_mode="event",
        event_seed=42,
        seeded_variation=False,
        context_buffer=int(B),
    )
    test_ds = train_loader.dataset.dataset
    train_idx = np.asarray(train_loader.dataset.indices)
    val_idx = np.asarray(val_loader.dataset.indices)
    test_idx = np.asarray(test_loader.dataset.indices)
    Xtr, ytr = _window_feature_matrix(test_ds, train_idx)
    Xva, yva = _window_feature_matrix(test_ds, val_idx)
    Xte, yte = _window_feature_matrix(test_ds, test_idx)
    pi = float(yte.mean()) if len(yte) else 0.0
    trivial_f1 = 2 * pi / (1 + pi) if pi > 0 else 0.0
    meta = {
        "B": int(B),
        "event_seed": 42,
        "seeded_variation": False,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "n_train_pos": int(ytr.sum()),
        "n_val_pos": int(yva.sum()),
        "n_test_pos": int(yte.sum()),
        "train_pos_rate": float(ytr.mean()) if len(ytr) else 0.0,
        "val_pos_rate": float(yva.mean()) if len(yva) else 0.0,
        "test_base_rate": pi,
        "trivial_f1": trivial_f1,
        "n_features": int(Xtr.shape[1]) if len(Xtr) else 0,
    }
    return Xtr, ytr, Xva, yva, Xte, yte, meta


def _eval_isolation_forest(
    Xtr, ytr, Xva, yva, Xte, yte, trivial_f1, pi
) -> Dict:
    """Fit on train *normal* windows only; anomaly score = -score_samples."""
    normal = ytr == 0
    if normal.sum() < 10:
        raise RuntimeError("Too few normal train windows for IsolationForest")

    grid = []
    for n_estimators in (100, 200, 500):
        for max_samples in ("auto", 0.5, 1.0):
            grid.append({"n_estimators": n_estimators, "max_samples": max_samples})

    best = None
    for params in grid:
        clf = IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr[normal])
        # Higher score => more anomalous
        sva_raw = -clf.score_samples(Xva)
        ste_raw = -clf.score_samples(Xte)
        # Map to [0,1] using validation range so a shared threshold grid applies
        lo, hi = float(sva_raw.min()), float(sva_raw.max())
        if hi - lo < 1e-12:
            sva = np.zeros_like(sva_raw)
            ste = np.zeros_like(ste_raw)
        else:
            sva = (sva_raw - lo) / (hi - lo)
            ste = (ste_raw - lo) / (hi - lo)
        thresh, val_f1 = _best_threshold(yva, sva)
        cand = (val_f1, params, thresh, ste, sva)
        if best is None or cand[0] > best[0]:
            best = cand

    val_f1, params, thresh, ste, _ = best
    row = _metrics_from_scores(
        "IsolationForest",
        yte,
        ste,
        thresh,
        trivial_f1,
        pi,
        extra={"params": params, "best_val_f1": float(val_f1), "fit_on": "train_normals"},
    )
    return row


def _pca_recon_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    Xs = model.named_steps["scaler"].transform(X)
    Z = model.named_steps["pca"].transform(Xs)
    Xhat = model.named_steps["pca"].inverse_transform(Z)
    err = np.mean((Xs - Xhat) ** 2, axis=1)
    return err


def _eval_pca(Xtr, ytr, Xva, yva, Xte, yte, trivial_f1, pi) -> Dict:
    """PCA reconstruction error; fit scaler+PCA on train normals only."""
    normal = ytr == 0
    if normal.sum() < 10:
        raise RuntimeError("Too few normal train windows for PCA")

    n_feat = Xtr.shape[1]
    max_comp = min(n_feat, int(normal.sum()) - 1)
    candidates = []
    for k in (5, 10, 20, 32, 50, 64):
        if 1 <= k <= max_comp:
            candidates.append(k)
    # Also try variance targets expressed as int components via PCA(n_components=float)
    var_targets = [0.90, 0.95, 0.99]

    best = None
    for n_comp in candidates:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp, random_state=42)),
            ]
        )
        model.fit(Xtr[normal])
        sva_raw = _pca_recon_scores(model, Xva)
        ste_raw = _pca_recon_scores(model, Xte)
        lo, hi = float(sva_raw.min()), float(sva_raw.max())
        if hi - lo < 1e-12:
            sva = np.zeros_like(sva_raw)
            ste = np.zeros_like(ste_raw)
        else:
            sva = (sva_raw - lo) / (hi - lo)
            ste = (ste_raw - lo) / (hi - lo)
        thresh, val_f1 = _best_threshold(yva, sva)
        params = {"n_components": n_comp}
        cand = (val_f1, params, thresh, ste)
        if best is None or cand[0] > best[0]:
            best = cand

    for var in var_targets:
        # sklearn allows float in (0,1] for variance explained
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=var, svd_solver="full", random_state=42)),
            ]
        )
        model.fit(Xtr[normal])
        n_comp = int(model.named_steps["pca"].n_components_)
        sva_raw = _pca_recon_scores(model, Xva)
        ste_raw = _pca_recon_scores(model, Xte)
        lo, hi = float(sva_raw.min()), float(sva_raw.max())
        if hi - lo < 1e-12:
            sva = np.zeros_like(sva_raw)
            ste = np.zeros_like(ste_raw)
        else:
            sva = (sva_raw - lo) / (hi - lo)
            ste = (ste_raw - lo) / (hi - lo)
        thresh, val_f1 = _best_threshold(yva, sva)
        params = {"n_components": var, "n_components_realized": n_comp}
        cand = (val_f1, params, thresh, ste)
        if best is None or cand[0] > best[0]:
            best = cand

    val_f1, params, thresh, ste = best
    row = _metrics_from_scores(
        "PCA-recon",
        yte,
        ste,
        thresh,
        trivial_f1,
        pi,
        extra={"params": params, "best_val_f1": float(val_f1), "fit_on": "train_normals"},
    )
    return row


def _eval_histgb(Xtr, ytr, Xva, yva, Xte, yte, trivial_f1, pi) -> Dict:
    clf = HistGradientBoostingClassifier(**HISTGB_PARAMS)
    clf.fit(Xtr, ytr)
    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    thresh, val_f1 = _best_threshold(yva, pva)
    return _metrics_from_scores(
        "HistGradientBoosting",
        yte,
        pte,
        thresh,
        trivial_f1,
        pi,
        extra={"params": HISTGB_PARAMS, "best_val_f1": float(val_f1), "fit_on": "all_train"},
    )


def run_for_B(B: int, with_histgb: bool = True) -> Dict:
    print(f"\n=== B={B} (event_seed=42, longest-first) ===")
    Xtr, ytr, Xva, yva, Xte, yte, meta = _load_split(B)
    print(
        f"  n_train/val/test={meta['n_train']}/{meta['n_val']}/{meta['n_test']} | "
        f"pi={meta['test_base_rate']:.4f} | trivial_F1={meta['trivial_f1']:.4f}"
    )
    rows: List[Dict] = []
    if_row = _eval_isolation_forest(Xtr, ytr, Xva, yva, Xte, yte, meta["trivial_f1"], meta["test_base_rate"])
    rows.append(if_row)
    print(
        f"  IsolationForest F1={if_row['test_f1']:.4f} "
        f"(dF1 {if_row['delta_f1_vs_trivial']:+.4f}) AUROC={if_row['test_auroc']:.4f} "
        f"params={if_row['params']}"
    )
    pca_row = _eval_pca(Xtr, ytr, Xva, yva, Xte, yte, meta["trivial_f1"], meta["test_base_rate"])
    rows.append(pca_row)
    print(
        f"  PCA-recon       F1={pca_row['test_f1']:.4f} "
        f"(dF1 {pca_row['delta_f1_vs_trivial']:+.4f}) AUROC={pca_row['test_auroc']:.4f} "
        f"params={pca_row['params']}"
    )
    if with_histgb:
        hg = _eval_histgb(Xtr, ytr, Xva, yva, Xte, yte, meta["trivial_f1"], meta["test_base_rate"])
        rows.append(hg)
        print(
            f"  HistGB          F1={hg['test_f1']:.4f} "
            f"(dF1 {hg['delta_f1_vs_trivial']:+.4f}) AUROC={hg['test_auroc']:.4f}"
        )
    return {"meta": meta, "models": rows}


def write_txt(payload: Dict) -> None:
    lines = []
    lines.append("MSL unsupervised references + B sweep (event_seed=42, longest-first)")
    lines.append("=" * 72)
    prim = payload["primary_B200"]
    lines.append(f"Primary B=200: pi={prim['meta']['test_base_rate']:.6f} "
                 f"trivial_F1={prim['meta']['trivial_f1']:.6f} "
                 f"n_test={prim['meta']['n_test']}")
    for m in prim["models"]:
        lines.append(
            f"  {m['name']:<22} F1={m['test_f1']:.4f} dF1={m['delta_f1_vs_trivial']:+.4f} "
            f"AUROC={m['test_auroc']:.4f} P={m['test_precision']:.4f} R={m['test_recall']:.4f} "
            f"thresh={m['best_thresh']:.2f} params={m.get('params')}"
        )
    lines.append("")
    lines.append("B sweep summary")
    lines.append("-" * 72)
    lines.append(
        f"{'B':>5} {'n_te':>5} {'pi':>7} {'trivF1':>7} "
        f"{'IF_F1':>7} {'IF_dF1':>8} {'IF_AUC':>7} "
        f"{'PCA_F1':>7} {'PCA_dF1':>8} {'PCA_AUC':>7} "
        f"{'HG_F1':>7} {'HG_dF1':>8}"
    )
    for block in payload["B_sweep"]:
        meta = block["meta"]
        by = {m["name"]: m for m in block["models"]}
        if_m = by["IsolationForest"]
        pca_m = by["PCA-recon"]
        hg_m = by.get("HistGradientBoosting")
        lines.append(
            f"{meta['B']:5d} {meta['n_test']:5d} {meta['test_base_rate']:7.4f} {meta['trivial_f1']:7.4f} "
            f"{if_m['test_f1']:7.4f} {if_m['delta_f1_vs_trivial']:+8.4f} {if_m['test_auroc']:7.4f} "
            f"{pca_m['test_f1']:7.4f} {pca_m['delta_f1_vs_trivial']:+8.4f} {pca_m['test_auroc']:7.4f} "
            f"{(hg_m['test_f1'] if hg_m else float('nan')):7.4f} "
            f"{(hg_m['delta_f1_vs_trivial'] if hg_m else float('nan')):+8.4f}"
        )
    # LaTeX-friendly rows
    lines.append("")
    lines.append("LaTeX rows (B sweep)")
    for block in payload["B_sweep"]:
        meta = block["meta"]
        by = {m["name"]: m for m in block["models"]}
        if_m, pca_m = by["IsolationForest"], by["PCA-recon"]
        hg_m = by.get("HistGradientBoosting")
        lines.append(
            f"{meta['B']} & {meta['n_test']} & {meta['test_base_rate']:.4f} & "
            f"{meta['trivial_f1']:.4f} & "
            f"{if_m['test_f1']:.4f} & {if_m['test_auroc']:.3f} & "
            f"{pca_m['test_f1']:.4f} & {pca_m['test_auroc']:.3f} & "
            f"{hg_m['test_f1']:.4f} \\\\"
            if hg_m
            else ""
        )
    text = "\n".join(lines) + "\n"
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote {RESULTS_TXT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--B-values",
        type=int,
        nargs="+",
        default=DEFAULT_B_VALUES,
        help="Context buffer values to sweep",
    )
    ap.add_argument("--skip-histgb", action="store_true")
    args = ap.parse_args()

    b_values = list(dict.fromkeys(int(b) for b in args.B_values))
    if PRIMARY_B not in b_values:
        b_values = [PRIMARY_B] + b_values

    sweep = []
    primary = None
    for B in b_values:
        block = run_for_B(B, with_histgb=not args.skip_histgb)
        sweep.append(block)
        if B == PRIMARY_B:
            primary = block

    payload = {
        "meta": {
            "dataset": "MSL",
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "event_seed": 42,
            "seeded_variation": False,
            "assignment": "longest-first",
            "feature": "5S window stats (mean,std,min,max,slope)",
            "B_values": b_values,
            "primary_B": PRIMARY_B,
            "histgb_params": HISTGB_PARAMS,
        },
        "primary_B200": primary,
        "B_sweep": sweep,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")
    write_txt(payload)


if __name__ == "__main__":
    main()
