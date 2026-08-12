"""
Revision experiments for Internet of Things major-revision comments.

Phases:
  verify   — T', epsilon/scaler, MSL channel counts, split rule, SMD SD (no training)
  shallow  — logistic regression + HistGB on window stats (same event split seed=42)
  point    — point-level P/R/F1 for Standard Transformer and CNN-only (event_seed=42)
  event    — multi event-split variance (seeded_variation=True)
  smd_auroc — re-run SMD LST-TFDN with AUROC/AUPRC (resume-safe)
  figures  — regenerate three figures with trivial F1=0.6455 overlay
  all      — verify → shallow → point → event → figures (smd_auroc optional)

Resume-safe JSON: revision_experiment_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import (
    TimeSeriesDataset,
    _anomaly_segments,
    _assign_segments_by_duration,
    _event_level_indices,
    get_dataloaders,
    load_dataset,
)
from model import (
    build_cnn_only_baseline,
    build_lst_tfdn_baseline,
    build_standard_transformer_baseline,
)
from phase1_msl_unified import (
    BATCH_SIZE,
    DEFAULT_POS_WEIGHT,
    EPOCHS,
    STRIDE,
    WINDOW_SIZE,
    build_model,
    load_msl_event_meta,
    point_level_metrics,
    set_seed,
    train_one,
)
from sklearn.preprocessing import MinMaxScaler

RESULTS_JSON = "revision_experiment_results.json"
VERIFY_JSON = "revision_verify_report.json"

EVENT_SEEDS = [42, 123, 456, 789, 1024]
INIT_SEEDS_EVENT = [42, 123, 456]  # crossed with event seeds
INIT_SEEDS_POINT = [42, 123, 456, 789, 1024]
MODELS_CORE = ["LST-TFDN", "Standard Transformer", "CNN-only"]

TRIVIAL_F1_MSL = 2 * (284 / 596) / (1 + 284 / 596)  # 0.6455...


def _load_results() -> Dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {
        "shallow": [],
        "point": [],
        "event_split": [],
        "smd_auroc": [],
        "meta": {},
    }


def _save_results(data: Dict) -> None:
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


# ---------------------------------------------------------------------------
# Phase: verify
# ---------------------------------------------------------------------------
def phase_verify() -> Dict:
    report: Dict = {}
    base = "datasets/MSL"
    train, _ = load_dataset(os.path.join(base, "MSL_train.npy"))
    test, labels = load_dataset(
        os.path.join(base, "MSL_test.npy"),
        os.path.join(base, "MSL_test_label.npy"),
    )
    labels = labels.astype(int).ravel()

    # T'
    import torch.nn as nn

    x = torch.randn(1, train.shape[1], WINDOW_SIZE)
    cnn = nn.Sequential(
        nn.Conv1d(train.shape[1], 32, 5, 1, 2),
        nn.ReLU(),
        nn.Conv1d(32, 32, 5, 2, 2),
    )
    t_prime = int(cnn(x).shape[-1])
    report["T_prime_actual"] = t_prime
    report["T_prime_paper_formula_floor_T_over_2_plus_1"] = WINDOW_SIZE // 2 + 1
    report["T_prime_recommendation"] = (
        f"Use T'={t_prime} everywhere; PE buffer in model.py used window_size//2+1="
        f"{WINDOW_SIZE // 2 + 1} (oversized by one)."
    )

    # Min-max epsilon: sklearn MinMaxScaler has no epsilon; constant channels -> 0 range
    scaler = MinMaxScaler()
    scaler.fit(train)
    report["minmax_epsilon"] = (
        "Not used in code. sklearn.preprocessing.MinMaxScaler with default "
        "feature_range=(0,1); constant channels keep scale_=1 and min_ handling "
        "per sklearn (zero-variance channels map to 0 after transform)."
    )
    report["scaler_scale_zeros"] = int(np.sum(scaler.scale_ == 0))

    # Channels
    n_bin = n_cont = 0
    cont_idx = []
    for i in range(train.shape[1]):
        u = np.unique(train[:, i])
        rounded = set(np.round(u.astype(float), 6).tolist())
        if len(u) <= 3 and rounded.issubset({0.0, 1.0}):
            n_bin += 1
        else:
            n_cont += 1
            cont_idx.append(i)
    report["msl_channels"] = {
        "total": int(train.shape[1]),
        "continuous": n_cont,
        "binary_onehot_like": n_bin,
        "continuous_indices": cont_idx,
    }

    # Split rule: legacy seed=42 vs seeded variation
    segs = _anomaly_segments(labels)
    legacy = {}
    varied = {}
    for seed in EVENT_SEEDS:
        tr, va, te = _event_level_indices(
            labels, WINDOW_SIZE, STRIDE, 200, seed=seed, seeded_variation=False
        )
        legacy[seed] = {
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "test_hash": hash(te.tobytes()) % 10**9,
        }
        tr2, va2, te2 = _event_level_indices(
            labels, WINDOW_SIZE, STRIDE, 200, seed=seed, seeded_variation=True
        )
        varied[seed] = {
            "n_train": len(tr2),
            "n_val": len(va2),
            "n_test": len(te2),
            "test_hash": hash(te2.tobytes()) % 10**9,
            "test_pos_rate": float(
                np.mean(
                    [
                        labels[wi * STRIDE : wi * STRIDE + WINDOW_SIZE].max()
                        for wi in te2
                    ]
                )
            )
            if len(te2)
            else None,
        }
    report["event_split_legacy_identical"] = len({v["test_hash"] for v in legacy.values()}) == 1
    report["event_split_seeded_n_unique"] = len({v["test_hash"] for v in varied.values()})
    report["event_split_legacy"] = legacy
    report["event_split_seeded_variation"] = varied
    report["split_allocator_rule"] = (
        "Greedy duration balance 50/25/25; longest-first. "
        "Legacy: seed does not change partition. "
        "seeded_variation=True: seeded order among equal lengths + random tie-break."
    )

    # Trivial F1
    pi = 284 / 596
    report["msl_test_base_rate"] = pi
    report["trivial_always_positive_f1"] = 2 * pi / (1 + pi)

    # SMD SD reconciliation
    smd_path = "smd_event_benchmark_results.json"
    if os.path.exists(smd_path):
        with open(smd_path, encoding="utf-8") as f:
            smd = json.load(f)["runs"]
        for name in sorted({r["name"] for r in smd}):
            f1s = [r["test_f1"] for r in smd if r["name"] == name]
            by_m: Dict[str, List[float]] = defaultdict(list)
            for r in smd:
                if r["name"] == name:
                    by_m[r["machine"]].append(r["test_f1"])
            means = [float(np.mean(v)) for v in by_m.values()]
            report.setdefault("smd_sd", {})[name] = {
                "pooled_n": len(f1s),
                "pooled_mean": float(np.mean(f1s)),
                "pooled_sd": float(np.std(f1s, ddof=1)),
                "machine_n": len(means),
                "machine_mean": float(np.mean(means)),
                "machine_sd": float(np.std(means, ddof=1)),
            }

    # Hardware note
    report["hardware"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "flop_counter_note": (
            "phase1 uses torch.utils.flop_counter.FlopCounterMode; "
            "PyTorch counts multiply-adds as FLOPs (MAC≈FLOP/2 if converting)."
        ),
    }

    with open(VERIFY_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {VERIFY_JSON}")
    return report


# ---------------------------------------------------------------------------
# Phase: shallow baselines
# ---------------------------------------------------------------------------
def _window_feature_matrix(dataset: TimeSeriesDataset, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sensor mean, std, min, max, slope over each window."""
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


def phase_shallow(data: Dict) -> Dict:
    print("\n=== SHALLOW BASELINES (event_seed=42, legacy split) ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        "MSL",
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split_mode="event",
        event_seed=42,
        seeded_variation=False,
    )
    # Recover underlying dataset + indices
    test_ds = train_loader.dataset.dataset  # Subset -> TimeSeriesDataset on test series
    train_idx = np.asarray(train_loader.dataset.indices)
    val_idx = np.asarray(val_loader.dataset.indices)
    test_idx = np.asarray(test_loader.dataset.indices)

    Xtr, ytr = _window_feature_matrix(test_ds, train_idx)
    Xva, yva = _window_feature_matrix(test_ds, val_idx)
    Xte, yte = _window_feature_matrix(test_ds, test_idx)
    pi = float(yte.mean())
    trivial_f1 = 2 * pi / (1 + pi) if pi > 0 else 0.0

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
    }

    rows = []
    for name, clf in models.items():
        print(f"  Fitting {name}...")
        clf.fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            pva = clf.predict_proba(Xva)[:, 1]
            pte = clf.predict_proba(Xte)[:, 1]
        else:
            pva = clf.decision_function(Xva)
            pte = clf.decision_function(Xte)
            pva = 1 / (1 + np.exp(-pva))
            pte = 1 / (1 + np.exp(-pte))

        # threshold on validation F1
        best_t, best_f = 0.5, -1.0
        for t in np.arange(0.01, 0.99, 0.01):
            f = f1_score(yva, (pva >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        pred = (pte >= best_t).astype(int)
        row = {
            "name": name,
            "event_seed": 42,
            "test_base_rate": pi,
            "trivial_f1": trivial_f1,
            "best_thresh": best_t,
            "best_val_f1": float(best_f),
            "test_f1": float(f1_score(yte, pred, zero_division=0)),
            "test_precision": float(precision_score(yte, pred, zero_division=0)),
            "test_recall": float(recall_score(yte, pred, zero_division=0)),
            "test_auroc": safe_auroc(yte, pte),
            "test_auprc": safe_auprc(yte, pte),
        }
        row["delta_f1_vs_trivial"] = row["test_f1"] - trivial_f1
        rows.append(row)
        print(
            f"    F1={row['test_f1']:.4f} (dF1 vs trivial {row['delta_f1_vs_trivial']:+.4f}) | "
            f"AUROC={row['test_auroc']:.4f} | thresh={best_t:.2f}"
        )

    # Analytic references
    rows.insert(
        0,
        {
            "name": "Always-positive",
            "test_f1": trivial_f1,
            "test_precision": pi,
            "test_recall": 1.0,
            "test_auroc": 0.5,
            "test_auprc": pi,
            "delta_f1_vs_trivial": 0.0,
        },
    )
    data["shallow"] = rows
    _save_results(data)
    return data


# ---------------------------------------------------------------------------
# Phase: point-level for StdT + CNN-only
# ---------------------------------------------------------------------------
def phase_point(data: Dict, epochs: int = EPOCHS) -> Dict:
    print("\n=== POINT-LEVEL METRICS (StdT + CNN-only, event_seed=42) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(
        "MSL",
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split_mode="event",
        event_seed=42,
        seeded_variation=False,
    )
    num_sensors = next(iter(train_loader))[0].shape[1]
    _, point_labels, _, _, test_window_idx = load_msl_event_meta()

    done = {(r["name"], r["seed"]) for r in data.get("point", [])}
    # Also reuse LST-TFDN point metrics from phase1 if present
    if os.path.exists("phase1_msl_unified_results.json"):
        with open("phase1_msl_unified_results.json", encoding="utf-8") as f:
            p1 = json.load(f)
        for r in p1.get("runs", []):
            if (
                r["name"] == "LST-TFDN"
                and abs(r.get("pos_weight", 10) - 10) < 1e-9
                and r.get("point_f1") is not None
            ):
                key = ("LST-TFDN", r["seed"])
                if key not in done:
                    data.setdefault("point", []).append(
                        {
                            "name": "LST-TFDN",
                            "seed": r["seed"],
                            "source": "phase1_msl_unified_results.json",
                            "test_f1": r["test_f1"],
                            "point_f1": r["point_f1"],
                            "point_precision": r["point_precision"],
                            "point_recall": r["point_recall"],
                            "point_auroc": r.get("point_auroc"),
                            "point_auprc": r.get("point_auprc"),
                        }
                    )
                    done.add(key)
        _save_results(data)

    jobs = [
        (name, seed)
        for name in ["Standard Transformer", "CNN-only"]
        for seed in INIT_SEEDS_POINT
    ]
    for i, (name, seed) in enumerate(jobs, 1):
        if (name, seed) in done:
            print(f"[{i}/{len(jobs)}] SKIP {name} seed={seed}")
            continue
        print(f"[{i}/{len(jobs)}] TRAIN {name} seed={seed}")
        set_seed(seed)
        model = build_model(name, num_sensors).to(device)
        t0 = time.time()
        metrics = train_one(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs,
            DEFAULT_POS_WEIGHT,
            verbose=False,
        )
        point = point_level_metrics(
            np.asarray(metrics["test_probs"]),
            test_window_idx,
            point_labels,
            metrics["best_thresh"],
        )
        row = {
            "name": name,
            "seed": seed,
            "seconds": round(time.time() - t0, 1),
            "test_f1": metrics["test_f1"],
            "test_auroc": metrics["test_auroc"],
            "best_thresh": metrics["best_thresh"],
            **point,
        }
        print(
            f"    window F1={row['test_f1']:.4f} | point F1={row['point_f1']:.4f} "
            f"| P={row['point_precision']:.4f} R={row['point_recall']:.4f} "
            f"| {row['seconds']}s"
        )
        data.setdefault("point", []).append(row)
        done.add((name, seed))
        _save_results(data)

    # Summary
    for name in ["LST-TFDN", "Standard Transformer", "CNN-only"]:
        rows = [r for r in data["point"] if r["name"] == name]
        if not rows:
            continue
        pf = [r["point_f1"] for r in rows]
        print(
            f"SUMMARY {name}: point F1 {np.mean(pf):.4f}±{np.std(pf, ddof=1) if len(pf)>1 else 0:.4f} "
            f"(n={len(pf)})"
        )
    return data


# ---------------------------------------------------------------------------
# Phase: multi event-split variance
# ---------------------------------------------------------------------------
def phase_event(data: Dict, epochs: int = EPOCHS) -> Dict:
    print("\n=== EVENT-SPLIT VARIANCE (seeded_variation=True) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    done = {
        (r["name"], r["event_seed"], r["init_seed"])
        for r in data.get("event_split", [])
    }
    jobs = [
        (name, es, ins)
        for es in EVENT_SEEDS
        for ins in INIT_SEEDS_EVENT
        for name in MODELS_CORE
    ]
    total = len(jobs)
    for i, (name, event_seed, init_seed) in enumerate(jobs, 1):
        key = (name, event_seed, init_seed)
        if key in done:
            print(f"[{i}/{total}] SKIP {key}")
            continue
        print(f"[{i}/{total}] TRAIN {name} event_seed={event_seed} init={init_seed}")
        train_loader, val_loader, test_loader = get_dataloaders(
            "MSL",
            batch_size=BATCH_SIZE,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            split_mode="event",
            event_seed=event_seed,
            seeded_variation=True,
        )
        # Skip degenerate empty splits
        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print("    SKIP empty split")
            continue
        num_sensors = next(iter(train_loader))[0].shape[1]
        y_te = []
        for _, t in test_loader:
            y_te.extend(t.numpy().tolist())
        y_te = np.asarray(y_te)
        pi = float(y_te.mean()) if len(y_te) else float("nan")
        trivial = 2 * pi / (1 + pi) if pi and pi > 0 else 0.0

        set_seed(init_seed)
        model = build_model(name, num_sensors).to(device)
        t0 = time.time()
        metrics = train_one(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs,
            DEFAULT_POS_WEIGHT,
            verbose=False,
        )
        row = {
            "name": name,
            "event_seed": event_seed,
            "init_seed": init_seed,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
            "test_base_rate": pi,
            "trivial_f1": trivial,
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_auroc": metrics["test_auroc"],
            "test_auprc": metrics["test_auprc"],
            "best_thresh": metrics["best_thresh"],
            "delta_f1_vs_trivial": metrics["test_f1"] - trivial,
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"    F1={row['test_f1']:.4f} dF1={row['delta_f1_vs_trivial']:+.4f} "
            f"AUROC={row['test_auroc']:.4f} pi={pi:.3f} ({row['seconds']}s)"
        )
        data.setdefault("event_split", []).append(row)
        done.add(key)
        _save_results(data)

    # Aggregate: mean over init seeds per event seed, then SD across event seeds
    print("\n--- Event-split summary (mean over init seeds, then across event seeds) ---")
    for name in MODELS_CORE:
        by_es: Dict[int, List[float]] = defaultdict(list)
        for r in data["event_split"]:
            if r["name"] == name:
                by_es[r["event_seed"]].append(r["test_f1"])
        if not by_es:
            continue
        per_es = [float(np.mean(v)) for v in by_es.values()]
        print(
            f"{name}: across {len(per_es)} event seeds → "
            f"F1 {np.mean(per_es):.4f}±{np.std(per_es, ddof=1) if len(per_es)>1 else 0:.4f} "
            f"(per-event means: {[round(x,4) for x in per_es]})"
        )
    return data


# ---------------------------------------------------------------------------
# Phase: SMD AUROC/AUPRC for LST-TFDN
# ---------------------------------------------------------------------------
def phase_smd_auroc(data: Dict, epochs: int = EPOCHS, max_machines: Optional[int] = None) -> Dict:
    """Re-evaluate SMD machines with AUROC/AUPRC (LST-TFDN, 3 seeds)."""
    print("\n=== SMD AUROC/AUPRC (LST-TFDN) ===")
    from baselines_event import train_and_evaluate as _tae
    from benchmark_smd_all import list_smd_machines
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Patch: wrap training to also return AUROC — reimplement thin wrapper
    from baselines_event import (
        BATCH_SIZE as B,
        DEFAULT_SEEDS,
        POS_WEIGHT,
        STRIDE as S,
        WINDOW_SIZE as W,
        count_params,
        set_seed as ss,
    )
    from model import build_lst_tfdn_baseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machines = list_smd_machines()
    if max_machines:
        machines = machines[:max_machines]
    done = {(r["machine"], r["seed"]) for r in data.get("smd_auroc", [])}
    seeds = DEFAULT_SEEDS  # [42,43,44]

    for mi, machine in enumerate(machines, 1):
        for seed in seeds:
            if (machine, seed) in done:
                continue
            print(f"[{mi}/{len(machines)}] {machine} seed={seed}")
            try:
                tr, va, te = get_dataloaders(
                    "SMD",
                    batch_size=B,
                    window_size=W,
                    stride=S,
                    split_mode="event",
                    machine_id=machine,
                    event_seed=42,
                    seeded_variation=False,
                )
            except Exception as e:
                print(f"  skip load: {e}")
                continue
            if len(tr.dataset) == 0 or len(va.dataset) == 0 or len(te.dataset) == 0:
                print("  skip empty")
                continue
            ns = next(iter(tr))[0].shape[1]
            ss(seed)
            model = build_lst_tfdn_baseline(ns, W).to(device)
            metrics = _tae(model, tr, va, te, device, epochs, verbose=False)
            # recompute AUROC from a fresh forward
            model.eval()
            probs, targets = [], []
            with torch.no_grad():
                for xb, yb in te:
                    probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
                    targets.extend(yb.numpy())
            probs = np.asarray(probs)
            targets = np.asarray(targets)
            auroc = safe_auroc(targets, probs)
            auprc = safe_auprc(targets, probs)
            pi = float(targets.mean()) if len(targets) else float("nan")
            trivial = 2 * pi / (1 + pi) if pi and pi > 0 else 0.0
            row = {
                "machine": machine,
                "seed": seed,
                "name": "LST-TFDN",
                "num_params": count_params(model),
                "test_f1": metrics["test_f1"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_auroc": auroc,
                "test_auprc": auprc,
                "test_base_rate": pi,
                "trivial_f1": trivial,
                "delta_f1_vs_trivial": metrics["test_f1"] - trivial,
                "best_thresh": metrics["best_thresh"],
            }
            print(
                f"    F1={row['test_f1']:.3f} AUROC={auroc:.3f} AUPRC={auprc:.3f} "
                f"dF1={row['delta_f1_vs_trivial']:+.3f}"
            )
            data.setdefault("smd_auroc", []).append(row)
            done.add((machine, seed))
            _save_results(data)

    # Per-machine means
    by_m: Dict[str, List[Dict]] = defaultdict(list)
    for r in data.get("smd_auroc", []):
        by_m[r["machine"]].append(r)
    print("\n--- SMD per-machine mean AUROC (LST-TFDN) ---")
    aurocs = []
    for m, rows in sorted(by_m.items()):
        a = float(np.mean([r["test_auroc"] for r in rows]))
        aurocs.append(a)
        print(f"  {m}: AUROC={a:.3f} F1={np.mean([r['test_f1'] for r in rows]):.3f}")
    if aurocs:
        print(f"Overall mean AUROC={np.mean(aurocs):.3f}±{np.std(aurocs, ddof=1):.3f}")
    return data


# ---------------------------------------------------------------------------
# Phase: figures with trivial overlay
# ---------------------------------------------------------------------------
def phase_figures() -> None:
    print("\n=== REGENERATE FIGURES WITH TRIVIAL OVERLAY ===")
    import matplotlib.pyplot as plt

    out_dir = os.path.join("output_images", "new_images")
    os.makedirs(out_dir, exist_ok=True)
    trivial = TRIVIAL_F1_MSL

    # 1) Main F1 comparison
    names = ["LST-TFDN", "Standard\nTransformer", "CNN-only"]
    f1 = [0.5290, 0.6315, 0.5188]
    std = [0.0242, 0.0328, 0.0354]
    colors = ["#1976D2", "#6A1B9A", "#00897B"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(names))
    ax.bar(x, f1, yerr=std, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.axhline(trivial, color="#C62828", linestyle="--", linewidth=1.8, label=f"Always-positive F1 = {trivial:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Window F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("MSL event-level holdout (5 init seeds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "results_main_f1_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # 2) Protocol ranking
    models = ["LST-TFDN", "Std Transformer", "CNN-only"]
    leaky = [0.9105, 0.9000, 0.8993]
    event = [0.5257, 0.6337, 0.5029]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, leaky, w, label="Leaky stratified", color="#90A4AE", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, event, w, label="Event-level", color="#1976D2", edgecolor="black", linewidth=0.5)
    ax.axhline(trivial, color="#C62828", linestyle="--", linewidth=1.8, label=f"Always-positive (event π) F1 = {trivial:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Window F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leaky vs event-level F1 on MSL (3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "results_protocol_ranking.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # 3) SMD per-machine with per-machine trivial
    # Prefer live JSON; fall back to means if needed
    smd_path = "smd_event_benchmark_results.json"
    with open(smd_path, encoding="utf-8") as f:
        runs = [r for r in json.load(f)["runs"] if r["name"] == "LST-TFDN (Proposed)"]
    by_m: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        by_m[r["machine"]].append(r)

    # Need pos ratio — from table in paper / recompute if present
    # Use mean F1; trivial from revision results if available else approximate from paper table
    PAPER_POS = {
        "machine-1-1": 0.59, "machine-1-2": 0.35, "machine-1-3": 0.27, "machine-1-4": 0.30,
        "machine-1-5": 0.23, "machine-1-6": 0.25, "machine-1-7": 0.28, "machine-1-8": 0.25,
        "machine-2-1": 0.26, "machine-2-2": 0.70, "machine-2-3": 0.35, "machine-2-4": 0.47,
        "machine-2-5": 0.31, "machine-2-6": 0.37, "machine-2-7": 0.22, "machine-2-9": 0.62,
        "machine-3-1": 0.24, "machine-3-10": 0.34, "machine-3-11": 0.42, "machine-3-2": 0.20,
        "machine-3-3": 0.23, "machine-3-4": 0.31, "machine-3-5": 0.49, "machine-3-6": 0.33,
        "machine-3-7": 0.24, "machine-3-8": 0.41, "machine-3-9": 0.25,
    }
    def _machine_key(m: str):
        parts = m.replace("machine-", "").split("-")
        return (int(parts[0]), int(parts[1]))

    # Natural numeric order (not lexicographic: 3-2 before 3-10).
    machines = sorted(by_m.keys(), key=_machine_key)
    f1m = [float(np.mean([r["test_f1"] for r in by_m[m]])) for m in machines]
    triv = []
    for m in machines:
        pi = PAPER_POS.get(m)
        triv.append(2 * pi / (1 + pi) if pi else np.nan)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(machines))
    colors = ["#C62828" if (not np.isnan(t) and f < t) else "#1976D2" for f, t in zip(f1m, triv)]
    ax.bar(x, f1m, color=colors, edgecolor="black", linewidth=0.4, zorder=2)
    # Markers + light stems only: do not connect categorical machines with a polyline.
    ax.scatter(
        x, triv, s=28, color="#C62828", marker="o", zorder=3,
        label=r"Per-machine trivial $2\pi_m/(1+\pi_m)$",
    )
    for xi, ti in zip(x, triv):
        if np.isnan(ti):
            continue
        ax.plot([xi, xi], [0, ti], color="#C62828", linestyle=":", linewidth=0.9, alpha=0.45, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("machine-", "") for m in machines], rotation=90, fontsize=7)
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("SMD per-machine LST-TFDN F1 vs machine-specific trivial baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "results_smd_per_machine_f1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Copy into elsevier folders if present
    for dest in [
        os.path.join("paper", "elsevier_iot", "new_images"),
        os.path.join("paper", "elsevier_iot", "comments", "LST-TFDN_IoT_Revision2", "new_images"),
    ]:
        if os.path.isdir(dest):
            for name in [
                "results_main_f1_comparison.png",
                "results_protocol_ranking.png",
                "results_smd_per_machine_f1.png",
            ]:
                src = os.path.join(out_dir, name)
                dst = os.path.join(dest, name)
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())
                print(f"  Copied -> {dst}")


def print_final_tables(data: Dict) -> None:
    print("\n" + "=" * 72)
    print("REVISION EXPERIMENT TABLES")
    print("=" * 72)
    if data.get("shallow"):
        print("\n[Shallow / trivial]")
        for r in data["shallow"]:
            print(
                f"  {r['name']:<28} F1={r.get('test_f1', float('nan')):.4f} "
                f"dF1={r.get('delta_f1_vs_trivial', float('nan')):+.4f} "
                f"AUROC={r.get('test_auroc', float('nan')):.4f}"
            )
    if data.get("point"):
        print("\n[Point-level]")
        by = defaultdict(list)
        for r in data["point"]:
            by[r["name"]].append(r)
        for name, rows in by.items():
            pf = [r["point_f1"] for r in rows]
            pp = [r["point_precision"] for r in rows]
            pr = [r["point_recall"] for r in rows]
            print(
                f"  {name:<28} point F1={np.mean(pf):.4f}±{np.std(pf, ddof=1) if len(pf)>1 else 0:.4f} "
                f"P={np.mean(pp):.4f} R={np.mean(pr):.4f} (n={len(rows)})"
            )
    if data.get("event_split"):
        print("\n[Event-split variance]")
        for name in MODELS_CORE:
            by_es = defaultdict(list)
            for r in data["event_split"]:
                if r["name"] == name:
                    by_es[r["event_seed"]].append(r["test_f1"])
            if not by_es:
                continue
            per = [float(np.mean(v)) for v in by_es.values()]
            print(
                f"  {name:<28} across-event F1={np.mean(per):.4f}±"
                f"{np.std(per, ddof=1) if len(per)>1 else 0:.4f} n_event={len(per)}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["verify", "shallow", "point", "event", "smd_auroc", "figures", "all", "summary"],
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-machines", type=int, default=None, help="Limit SMD machines (debug)")
    ap.add_argument("--skip-smd", action="store_true", help="Skip SMD AUROC in --phase all")
    args = ap.parse_args()

    data = _load_results()
    data.setdefault("meta", {})
    data["meta"]["epochs"] = args.epochs
    data["meta"]["trivial_f1_msl"] = TRIVIAL_F1_MSL

    if args.phase in ("verify", "all"):
        phase_verify()
    if args.phase in ("shallow", "all"):
        phase_shallow(data)
    if args.phase in ("point", "all"):
        phase_point(data, epochs=args.epochs)
    if args.phase in ("event", "all"):
        phase_event(data, epochs=args.epochs)
    if args.phase == "smd_auroc" or (args.phase == "all" and not args.skip_smd):
        phase_smd_auroc(data, epochs=args.epochs, max_machines=args.max_machines)
    if args.phase in ("figures", "all"):
        phase_figures()
    if args.phase == "summary":
        print_final_tables(data)
    else:
        print_final_tables(data)
        _save_results(data)
        print(f"\nResults saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
