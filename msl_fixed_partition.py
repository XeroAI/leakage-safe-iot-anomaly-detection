"""
Phase-1 unified MSL experiments for Scientific Reports revision.

Goals (peer-review):
  - One fixed event-level split + one fixed seed list for baselines AND ablations
  - Reconcile CNN-only / Full LST-TFDN so Table 4 and Table 7 share the same numbers
  - Window-level F1/P/R + AUROC/AUPRC
  - 95% CIs and paired t-tests
  - Batch=1 latency + MACs/FLOPs
  - Class-weight sweep w in {1,5,10,20}
  - Point-level P/R/F1 for LST-TFDN

Resume-safe: completed (model, seed, pos_weight) runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode

from data_loader import (
    TimeSeriesDataset,
    _event_level_indices,
    get_dataloaders,
    load_dataset,
)
from model import (
    LST_TFDN,
    build_cnn_only_baseline,
    build_lst_tfdn_baseline,
    build_standard_transformer_baseline,
)
from sklearn.preprocessing import MinMaxScaler
from train import find_best_threshold


# ---------------------------------------------------------------------------
# Fixed protocol (do not change without re-running everything)
# ---------------------------------------------------------------------------
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
WINDOW_SIZE = 100
STRIDE = 10
DEFAULT_POS_WEIGHT = 10.0
DATASET_NAME = "MSL"
SPLIT_MODE = "event"
EVENT_SEED = 42  # split realization fixed; model init uses SEEDS
SEEDS = [42, 123, 456, 789, 1024]
W_SWEEP = [1.0, 5.0, 10.0, 20.0]
RESULTS_JSON = "phase1_msl_unified_results.json"
RESULTS_TXT = "phase1_msl_unified_results.txt"

# Canonical model registry. CNN-only and Full are shared across baseline/ablation tables.
MODELS = [
    ("LST-TFDN", "baseline+ablation", build_lst_tfdn_baseline),
    ("Standard Transformer", "baseline", build_standard_transformer_baseline),
    ("CNN-only", "baseline+ablation", build_cnn_only_baseline),
    (
        "w/ Standard Attention",
        "ablation",
        lambda n, w: LST_TFDN(
            n, w, d_model=32, heads=4, dropout=0.2,
            use_linear_attn=False, use_fe=True, use_cnn=True, use_transformer=True,
        ),
    ),
    (
        "w/ Standard PE (No FE)",
        "ablation",
        lambda n, w: LST_TFDN(
            n, w, d_model=32, heads=4, dropout=0.2,
            use_linear_attn=True, use_fe=False, use_cnn=True, use_transformer=True,
        ),
    ),
    (
        "w/o 1D CNN Front-end",
        "ablation",
        lambda n, w: LST_TFDN(
            n, w, d_model=32, heads=4, dropout=0.2,
            use_linear_attn=True, use_fe=True, use_cnn=False, use_transformer=True,
        ),
    ),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def measure_batch_latency_ms(model: nn.Module, loader: DataLoader, device, batch_size: int, repeats: int = 30) -> float:
    """Mean forward latency (ms) for a fixed batch size after warm-up."""
    model.eval()
    sample, _ = next(iter(loader))
    x = sample[:1].to(device)
    if batch_size > 1:
        # tile to requested batch for fair measurement when loader uses 64
        reps = (batch_size + x.shape[0] - 1) // x.shape[0]
        x = x.repeat(reps, 1, 1)[:batch_size]
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def measure_flops(model: nn.Module, num_sensors: int, device) -> Dict[str, float]:
    model.eval()
    x = torch.zeros(1, num_sensors, WINDOW_SIZE, device=device)
    with FlopCounterMode(display=False) as fcm:
        model(x)
    flops = float(fcm.get_total_flops())
    # Conv/Linear FLOPs in PyTorch counter are multiply-adds counted as FLOPs;
    # report both FLOPs and MACs≈FLOPs/2 for tables that expect MACs.
    return {"flops": flops, "macs": flops / 2.0}


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mean_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, mean
    se = float(values.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, df=n - 1) * se)
    return mean, half, float(values.std(ddof=1))


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"t": float("nan"), "p": float("nan"), "n": float(len(a))}
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "n": float(len(a))}


def load_msl_event_meta():
    """Return scaled test series, labels, and event-split window indices."""
    base = f"datasets/{DATASET_NAME}"
    train_data, _ = load_dataset(os.path.join(base, "MSL_train.npy"))
    test_data, test_labels = load_dataset(
        os.path.join(base, "MSL_test.npy"),
        os.path.join(base, "MSL_test_label.npy"),
    )
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    test_data = scaler.transform(test_data)
    train_idx, val_idx, test_idx = _event_level_indices(
        test_labels,
        WINDOW_SIZE,
        STRIDE,
        context_buffer=max(WINDOW_SIZE, 200),
        seed=EVENT_SEED,
    )
    return test_data, test_labels.astype(int).ravel(), train_idx, val_idx, test_idx


def point_level_metrics(
    window_probs: np.ndarray,
    window_indices: np.ndarray,
    point_labels: np.ndarray,
    thresh: float,
) -> Dict[str, float]:
    """
    Map window scores onto points (max score over covering windows), then
    threshold. Evaluate only on points covered by at least one test window.
    """
    n = len(point_labels)
    score = np.full(n, -np.inf, dtype=float)
    covered = np.zeros(n, dtype=bool)
    for prob, wi in zip(window_probs, window_indices):
        a = int(wi) * STRIDE
        b = a + WINDOW_SIZE
        score[a:b] = np.maximum(score[a:b], float(prob))
        covered[a:b] = True

    mask = covered
    if mask.sum() == 0:
        return {
            "point_f1": float("nan"),
            "point_precision": float("nan"),
            "point_recall": float("nan"),
            "point_auroc": float("nan"),
            "point_auprc": float("nan"),
            "n_points": 0,
        }

    y = point_labels[mask]
    s = score[mask]
    pred = (s > thresh).astype(int)
    return {
        "point_f1": float(f1_score(y, pred, zero_division=0)),
        "point_precision": float(precision_score(y, pred, zero_division=0)),
        "point_recall": float(recall_score(y, pred, zero_division=0)),
        "point_auroc": safe_auroc(y, s),
        "point_auprc": safe_auprc(y, s),
        "n_points": int(mask.sum()),
        "n_pos_points": int(y.sum()),
    }


def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            p = torch.sigmoid(model(data)).cpu().numpy()
            probs.extend(p)
            targets.extend(target.numpy())
    return np.asarray(probs, dtype=float), np.asarray(targets, dtype=float)


def train_one(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs: int,
    pos_weight: float,
    verbose: bool = True,
) -> Dict:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_thresh = 0.5
    best_state = None

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_probs, val_targets = collect_probs(model, val_loader, device)
        epoch_thresh, val_f1 = find_best_threshold(val_targets, val_probs)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_thresh = epoch_thresh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"      Epoch {epoch + 1:02d}/{epochs} | Val F1 {val_f1:.4f} | τ={epoch_thresh:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_probs, test_targets = collect_probs(model, test_loader, device)
    test_preds = (test_probs > best_thresh).astype(int)

    return {
        "test_f1": float(f1_score(test_targets, test_preds, zero_division=0)),
        "test_precision": float(precision_score(test_targets, test_preds, zero_division=0)),
        "test_recall": float(recall_score(test_targets, test_preds, zero_division=0)),
        "test_auroc": safe_auroc(test_targets, test_probs),
        "test_auprc": safe_auprc(test_targets, test_probs),
        "best_val_f1": float(best_val_f1),
        "best_thresh": float(best_thresh),
        "test_probs": test_probs.tolist(),
        "test_targets": test_targets.tolist(),
    }


def run_key(name: str, seed: int, pos_weight: float) -> str:
    return f"{name}||seed={seed}||w={pos_weight}"


def load_results(path: str = RESULTS_JSON) -> Dict:
    if not os.path.exists(path):
        return {
            "protocol": {
                "dataset": DATASET_NAME,
                "split_mode": SPLIT_MODE,
                "event_seed": EVENT_SEED,
                "seeds": SEEDS,
                "epochs": EPOCHS,
                "window": WINDOW_SIZE,
                "stride": STRIDE,
            },
            "runs": [],
            "hardware": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(data: Dict, path: str = RESULTS_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def existing_keys(runs: List[Dict]) -> set:
    return {run_key(r["name"], r["seed"], r["pos_weight"]) for r in runs}


def build_model(name: str, num_sensors: int):
    for n, _role, fn in MODELS:
        if n == name:
            return fn(num_sensors, WINDOW_SIZE)
    raise KeyError(name)


def measure_hardware(num_sensors: int, device, test_loader) -> Dict:
    hw = {}
    for name, _role, fn in MODELS:
        if name not in ("LST-TFDN", "Standard Transformer", "CNN-only"):
            continue
        model = fn(num_sensors, WINDOW_SIZE).to(device)
        flops = measure_flops(model, num_sensors, device)
        lat1 = measure_batch_latency_ms(model, test_loader, device, batch_size=1)
        lat64 = measure_batch_latency_ms(model, test_loader, device, batch_size=64)
        hw[name] = {
            "num_params": count_params(model),
            "flops_per_window": flops["flops"],
            "macs_per_window": flops["macs"],
            "latency_ms_batch1": lat1,
            "latency_ms_batch64": lat64,
        }
        print(
            f"  HW {name}: params={hw[name]['num_params']:,} | "
            f"FLOPs={flops['flops']/1e6:.2f}M | "
            f"batch1={lat1:.2f}ms | batch64={lat64:.2f}ms"
        )
    return hw


def run_phase1(
    epochs: int = EPOCHS,
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    do_w_sweep: bool = True,
    do_hardware: bool = True,
    verbose: bool = True,
) -> Dict:
    seeds = seeds or SEEDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(
        f"Phase-1 unified MSL | split={SPLIT_MODE} (event_seed={EVENT_SEED}) | "
        f"epochs={epochs} | seeds={seeds} | device={device}"
    )
    print("=" * 100)

    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_NAME,
        BATCH_SIZE,
        WINDOW_SIZE,
        stride=STRIDE,
        split_mode=SPLIT_MODE,
        event_seed=EVENT_SEED,
    )
    num_sensors = next(iter(train_loader))[0].shape[1]
    _, point_labels, _, _, test_window_idx = load_msl_event_meta()

    data = load_results()
    data["protocol"]["seeds"] = seeds
    data["protocol"]["epochs"] = epochs
    done = existing_keys(data["runs"])

    if do_hardware and not data.get("hardware"):
        print("\nMeasuring FLOPs / batch=1 latency...")
        data["hardware"] = measure_hardware(num_sensors, device, test_loader)
        save_results(data)

    # Main + ablation models at w=10
    model_names = models or [n for n, _, _ in MODELS]
    jobs = [(name, seed, DEFAULT_POS_WEIGHT) for name in model_names for seed in seeds]

    # Class-weight sweep for LST-TFDN only
    if do_w_sweep:
        for w in W_SWEEP:
            if abs(w - DEFAULT_POS_WEIGHT) < 1e-9:
                continue
            for seed in seeds:
                jobs.append(("LST-TFDN", seed, float(w)))

    total = len(jobs)
    for i, (name, seed, w) in enumerate(jobs, 1):
        key = run_key(name, seed, w)
        if key in done:
            print(f"[{i}/{total}] SKIP {key}")
            continue

        print(f"\n[{i}/{total}] TRAIN {key}")
        set_seed(seed)
        model = build_model(name, num_sensors).to(device)
        n_params = count_params(model)
        metrics = train_one(
            model, train_loader, val_loader, test_loader, device, epochs, w, verbose=verbose
        )

        # Point-level for LST-TFDN at default w (and all w for completeness)
        point = {}
        if name == "LST-TFDN":
            point = point_level_metrics(
                np.asarray(metrics["test_probs"]),
                test_window_idx,
                point_labels,
                metrics["best_thresh"],
            )
            print(
                f"    Point-level F1={point['point_f1']:.4f} | "
                f"P={point['point_precision']:.4f} | R={point['point_recall']:.4f}"
            )

        # Drop bulky arrays from saved run (keep summary metrics)
        run = {
            "name": name,
            "seed": seed,
            "pos_weight": w,
            "num_params": n_params,
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_auroc": metrics["test_auroc"],
            "test_auprc": metrics["test_auprc"],
            "best_val_f1": metrics["best_val_f1"],
            "best_thresh": metrics["best_thresh"],
            **point,
        }
        print(
            f"    Window F1={run['test_f1']:.4f} | P={run['test_precision']:.4f} | "
            f"R={run['test_recall']:.4f} | AUROC={run['test_auroc']:.4f} | "
            f"AUPRC={run['test_auprc']:.4f}"
        )
        data["runs"].append(run)
        done.add(key)
        save_results(data)

    summary = build_summary(data)
    data["summary"] = summary
    save_results(data)
    write_paper_tables(data, RESULTS_TXT)
    print_paper_tables(data)
    return data


def rows_for(runs: List[Dict], name: str, pos_weight: float = DEFAULT_POS_WEIGHT) -> List[Dict]:
    return [r for r in runs if r["name"] == name and abs(r["pos_weight"] - pos_weight) < 1e-9]


def aggregate_model(runs: List[Dict], name: str, pos_weight: float = DEFAULT_POS_WEIGHT) -> Optional[Dict]:
    rows = rows_for(runs, name, pos_weight)
    if not rows:
        return None
    f1 = np.array([r["test_f1"] for r in rows])
    f1_mean, f1_ci, f1_std = mean_ci95(f1)
    out = {
        "name": name,
        "n": len(rows),
        "num_params": rows[0]["num_params"],
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "f1_ci95_half": f1_ci,
        "precision_mean": float(np.mean([r["test_precision"] for r in rows])),
        "recall_mean": float(np.mean([r["test_recall"] for r in rows])),
        "auroc_mean": float(np.nanmean([r["test_auroc"] for r in rows])),
        "auprc_mean": float(np.nanmean([r["test_auprc"] for r in rows])),
        "f1_values": f1.tolist(),
    }
    if "point_f1" in rows[0]:
        out["point_f1_mean"] = float(np.nanmean([r.get("point_f1", np.nan) for r in rows]))
        out["point_precision_mean"] = float(np.nanmean([r.get("point_precision", np.nan) for r in rows]))
        out["point_recall_mean"] = float(np.nanmean([r.get("point_recall", np.nan) for r in rows]))
    return out


def build_summary(data: Dict) -> Dict:
    runs = data["runs"]
    models = []
    for name, _, _ in MODELS:
        agg = aggregate_model(runs, name, DEFAULT_POS_WEIGHT)
        if agg:
            models.append(agg)

    lst = np.array(aggregate_model(runs, "LST-TFDN", DEFAULT_POS_WEIGHT)["f1_values"])
    tests = {}
    for other in ("Standard Transformer", "CNN-only"):
        agg = aggregate_model(runs, other, DEFAULT_POS_WEIGHT)
        if agg is None:
            continue
        tests[f"LST-TFDN_vs_{other}"] = paired_ttest(lst, np.array(agg["f1_values"]))

    w_sweep = []
    for w in W_SWEEP:
        agg = aggregate_model(runs, "LST-TFDN", w)
        if agg:
            w_sweep.append(
                {
                    "pos_weight": w,
                    "f1_mean": agg["f1_mean"],
                    "f1_std": agg["f1_std"],
                    "precision_mean": agg["precision_mean"],
                    "recall_mean": agg["recall_mean"],
                    "n": agg["n"],
                }
            )

    return {
        "models_w10": models,
        "paired_ttests_f1": tests,
        "w_sweep_lst_tfdn": w_sweep,
        "hardware": data.get("hardware", {}),
    }


def print_paper_tables(data: Dict) -> None:
    summary = data["summary"]
    print("\n" + "=" * 110)
    print("TABLE A — Unified baselines + ablations (MSL, event-level, same seeds, w=10)")
    print("=" * 110)
    print(
        f"{'Model':<28} {'F1 mean±std':>16} {'95% CI ±':>10} {'P':>8} {'R':>8} "
        f"{'AUROC':>8} {'AUPRC':>8} {'Params':>10}"
    )
    print("-" * 110)
    for m in summary["models_w10"]:
        f1 = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
        print(
            f"{m['name']:<28} {f1:>16} {m['f1_ci95_half']:>10.4f} "
            f"{m['precision_mean']:>8.4f} {m['recall_mean']:>8.4f} "
            f"{m['auroc_mean']:>8.4f} {m['auprc_mean']:>8.4f} {m['num_params']:>10,}"
        )

    print("\nPaired t-tests on window F1 (LST-TFDN vs ...):")
    for k, v in summary["paired_ttests_f1"].items():
        print(f"  {k}: t={v['t']:.4f}, p={v['p']:.4g}, n={int(v['n'])}")

    print("\nTABLE B — LST-TFDN class-weight sweep")
    print(f"{'w':>6} {'F1':>16} {'Precision':>10} {'Recall':>10} {'n':>4}")
    for row in summary["w_sweep_lst_tfdn"]:
        f1 = f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        print(
            f"{row['pos_weight']:>6.0f} {f1:>16} {row['precision_mean']:>10.4f} "
            f"{row['recall_mean']:>10.4f} {row['n']:>4}"
        )

    lst = aggregate_model(data["runs"], "LST-TFDN", DEFAULT_POS_WEIGHT)
    if lst and "point_f1_mean" in lst:
        print("\nTABLE C — LST-TFDN point-level metrics (max-score over covering windows)")
        print(
            f"  Point F1={lst['point_f1_mean']:.4f} | "
            f"P={lst['point_precision_mean']:.4f} | R={lst['point_recall_mean']:.4f}"
        )

    if summary.get("hardware"):
        print("\nTABLE D — Hardware / efficiency (CPU proxy)")
        for name, h in summary["hardware"].items():
            print(
                f"  {name}: params={h['num_params']:,} | "
                f"FLOPs/window={h['flops_per_window']/1e6:.2f}M | "
                f"MACs/window={h['macs_per_window']/1e6:.2f}M | "
                f"lat@1={h['latency_ms_batch1']:.2f}ms | "
                f"lat@64={h['latency_ms_batch64']:.2f}ms"
            )
    print("=" * 110)


def write_paper_tables(data: Dict, path: str = RESULTS_TXT) -> None:
    summary = data["summary"]
    lines = []
    lines.append("Phase-1 Unified MSL Results (event-level, fixed split + seeds)")
    lines.append(f"Seeds: {data['protocol']['seeds']}")
    lines.append(f"Event split seed: {data['protocol']['event_seed']}")
    lines.append("")
    lines.append("=== Models (w=10) ===")
    for m in summary["models_w10"]:
        lines.append(
            f"{m['name']}: F1={m['f1_mean']:.4f}±{m['f1_std']:.4f} "
            f"(95% CI ±{m['f1_ci95_half']:.4f}), "
            f"P={m['precision_mean']:.4f}, R={m['recall_mean']:.4f}, "
            f"AUROC={m['auroc_mean']:.4f}, AUPRC={m['auprc_mean']:.4f}, "
            f"Params={m['num_params']:,}"
        )
        if "point_f1_mean" in m:
            lines.append(
                f"  point-level: F1={m['point_f1_mean']:.4f}, "
                f"P={m['point_precision_mean']:.4f}, R={m['point_recall_mean']:.4f}"
            )
    lines.append("")
    lines.append("=== Paired t-tests (window F1) ===")
    for k, v in summary["paired_ttests_f1"].items():
        lines.append(f"{k}: t={v['t']:.4f}, p={v['p']:.6g}, n={int(v['n'])}")
    lines.append("")
    lines.append("=== w sweep (LST-TFDN) ===")
    for row in summary["w_sweep_lst_tfdn"]:
        lines.append(
            f"w={row['pos_weight']:.0f}: F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}, "
            f"P={row['precision_mean']:.4f}, R={row['recall_mean']:.4f}"
        )
    lines.append("")
    lines.append("=== Hardware ===")
    for name, h in summary.get("hardware", {}).items():
        lines.append(
            f"{name}: params={h['num_params']:,}, "
            f"FLOPs={h['flops_per_window']:.0f}, MACs={h['macs_per_window']:.0f}, "
            f"batch1_ms={h['latency_ms_batch1']:.3f}, batch64_ms={h['latency_ms_batch64']:.3f}"
        )
    lines.append("")
    lines.append("=== LaTeX rows (baselines / ablations, w=10) ===")
    for m in summary["models_w10"]:
        lines.append(
            f"{m['name']} & ${m['f1_mean']:.4f}\\pm{m['f1_std']:.4f}$ & "
            f"{m['precision_mean']:.4f} & {m['recall_mean']:.4f} & "
            f"{m['auroc_mean']:.4f} & {m['auprc_mean']:.4f} & "
            f"{m['num_params']:,} \\\\"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {path} and {RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-1 unified MSL experiments")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to train",
    )
    parser.add_argument("--no-w-sweep", action="store_true")
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 epochs, 2 seeds, LST+CNN only, no w-sweep",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild tables from existing JSON without training",
    )
    args = parser.parse_args()

    if args.summarize_only:
        data = load_results()
        data["summary"] = build_summary(data)
        save_results(data)
        write_paper_tables(data)
        print_paper_tables(data)
    elif args.quick:
        run_phase1(
            epochs=2,
            seeds=[42, 123],
            models=["LST-TFDN", "CNN-only"],
            do_w_sweep=False,
            do_hardware=True,
            verbose=True,
        )
    else:
        run_phase1(
            epochs=args.epochs,
            seeds=args.seeds,
            models=args.models,
            do_w_sweep=not args.no_w_sweep,
            do_hardware=not args.no_hardware,
            verbose=True,
        )
