"""
Multi-partition MSL training campaign under event-exclusive holdout.

Uses frozen partitions from msl_event_splits.py:
  * nested checkpoint selection (val_select) vs threshold calibration (val_calib)
  * buffer-defined and unrestricted-stream test evaluation
  * locked HistGB_5S_fixedHP baseline per partition
  * CNN-only and LST-TFDN over event partitions x init seeds
  * optional Standard Transformer

Resume-safe JSON + per-window score NPZs.

Usage:
  python msl_multipartition_train.py --freeze-only
  python msl_multipartition_train.py --smoke
  python msl_multipartition_train.py --phase all
  python msl_multipartition_train.py --phase deep --models CNN-only LST-TFDN
  python msl_multipartition_train.py --phase histgb
  python msl_multipartition_train.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from model import (
    build_cnn_only_baseline,
    build_lst_tfdn_baseline,
    build_standard_transformer_baseline,
)
from msl_event_splits import (
    EVENT_SEEDS_20,
    SPLITS_DIR,
    WINDOW_SIZE,
    STRIDE,
    build_scaled_dataset,
    freeze_all,
    load_partition,
)
from train import find_best_threshold

ROOT = "msl_multipartition"
SCORES_DIR = os.path.join(ROOT, "scores")
RESULTS_JSON = os.path.join(ROOT, "results.json")
RESULTS_TXT = os.path.join(ROOT, "results_summary.txt")
LOG_PATH = os.path.join(ROOT, "runner.log")

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
POS_WEIGHT = 10.0
INIT_SEEDS_5 = [42, 123, 456, 789, 1024]
INIT_SEEDS_3 = [42, 123, 456]

HISTGB_FIXED = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
    early_stopping=False,
)

BUILDERS = {
    "CNN-only": build_cnn_only_baseline,
    "LST-TFDN": build_lst_tfdn_baseline,
    "Standard Transformer": build_standard_transformer_baseline,
}


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    os.makedirs(ROOT, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def metrics_at(y, p, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > thr).astype(int)
    pi = float(y.mean()) if len(y) else float("nan")
    trivial = 2 * pi / (1 + pi) if pi > 0 else 0.0
    f1 = float(f1_score(y, pred, zero_division=0))
    auprc = safe_auprc(y, p)
    return {
        "f1": f1,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if len(y) else float("nan"),
        "auroc": safe_auroc(y, p),
        "auprc": auprc,
        "auprc_norm": (auprc / pi) if pi > 0 else float("nan"),
        "prevalence": pi,
        "trivial_f1": trivial,
        "delta_f1": f1 - trivial,
        "thresh": float(thr),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "tau_at_grid_floor": bool(abs(thr - 0.01) < 1e-12),
        "tau_at_grid_ceil": bool(abs(thr - 0.98) < 1e-12),
    }


def features_5s(dataset, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for wi in indices:
        x, y = dataset[int(wi)]
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        amin = x.min(axis=1)
        amax = x.max(axis=1)
        t = np.arange(x.shape[1], dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-8)
        slope = (x * t[None, :]).mean(axis=1)
        xs.append(np.concatenate([mean, std, amin, amax, slope]))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            p = torch.sigmoid(model(data)).view(-1).cpu().numpy()
            probs.append(p)
            targets.append(target.numpy())
    if not probs:
        return np.asarray([]), np.asarray([])
    return np.concatenate(probs), np.concatenate(targets)


def make_loader(ds, indices: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(ds, indices.tolist()), batch_size=BATCH_SIZE, shuffle=shuffle)


def load_results() -> Dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {
        "meta": {
            "protocol_version": "phaseB_v1",
            "epochs": EPOCHS,
            "pos_weight": POS_WEIGHT,
            "histgb": HISTGB_FIXED,
            "event_seeds": EVENT_SEEDS_20,
            "init_seeds_deep": INIT_SEEDS_5,
        },
        "deep_runs": [],
        "histgb_runs": [],
    }


def save_results(data: Dict) -> None:
    os.makedirs(ROOT, exist_ok=True)
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, RESULTS_JSON)


def deep_key(name: str, es: int, init: int) -> Tuple:
    return (name, es, init)


def train_deep_one(
    name: str,
    event_seed: int,
    init_seed: int,
    ds,
    part: Dict,
    device,
    epochs: int,
) -> Dict:
    idx = part["window_indices"]
    train_loader = make_loader(ds, idx["train"], shuffle=True)
    sel_loader = make_loader(ds, idx["val_select"], shuffle=False)
    cal_idx = idx["val_calib"]
    use_calib = len(cal_idx) > 0
    cal_loader = make_loader(ds, cal_idx if use_calib else idx["val_select"], shuffle=False)
    buf_loader = make_loader(ds, idx["test_buffer"], shuffle=False)
    unr_loader = make_loader(ds, idx["test_unrestricted"], shuffle=False)

    num_sensors = int(ds[0][0].shape[0])
    set_seed(init_seed)
    model = BUILDERS[name](num_sensors, WINDOW_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_sel_f1 = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = criterion(model(data).view(-1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sel_p, sel_y = collect_probs(model, sel_loader, device)
        _, sel_f1 = find_best_threshold(sel_y, sel_p)
        if sel_f1 > best_sel_f1:
            best_sel_f1 = float(sel_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    cal_p, cal_y = collect_probs(model, cal_loader, device)
    thr, cal_f1 = find_best_threshold(cal_y, cal_p)

    buf_p, buf_y = collect_probs(model, buf_loader, device)
    unr_p, unr_y = collect_probs(model, unr_loader, device)
    buf_m = metrics_at(buf_y, buf_p, thr)
    unr_m = metrics_at(unr_y, unr_p, thr)

    score_path = os.path.join(
        SCORES_DIR,
        f"deep_{name.replace(' ', '_')}_es{event_seed}_init{init_seed}.npz",
    )
    os.makedirs(SCORES_DIR, exist_ok=True)
    np.savez_compressed(
        score_path,
        probs_buffer=buf_p.astype(np.float32),
        targets_buffer=buf_y.astype(np.float32),
        window_indices_buffer=idx["test_buffer"],
        probs_unrestricted=unr_p.astype(np.float32),
        targets_unrestricted=unr_y.astype(np.float32),
        window_indices_unrestricted=idx["test_unrestricted"],
        probs_calib=cal_p.astype(np.float32),
        targets_calib=cal_y.astype(np.float32),
        thresh=np.asarray([thr], dtype=np.float32),
        used_calib_split=np.asarray([int(use_calib)]),
    )

    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    row = {
        "name": name,
        "event_seed": event_seed,
        "init_seed": init_seed,
        "n_params": n_params,
        "device": str(device),
        "epochs": epochs,
        "best_select_f1": best_sel_f1,
        "calib_f1": float(cal_f1),
        "used_calib_split": use_calib,
        "threshold": float(thr),
        "buffer": buf_m,
        "unrestricted": unr_m,
        "score_path": score_path,
        "seconds": round(time.time() - t0, 1),
        "partition_path": part.get("path"),
    }
    return row


def run_histgb_partition(event_seed: int, ds, part: Dict) -> Dict:
    idx = part["window_indices"]
    Xtr, ytr = features_5s(ds, idx["train"])
    # Threshold on calib if available else select (same rule as deep)
    cal_idx = idx["val_calib"] if len(idx["val_calib"]) else idx["val_select"]
    use_calib = len(idx["val_calib"]) > 0
    Xca, yca = features_5s(ds, cal_idx)
    Xbuf, ybuf = features_5s(ds, idx["test_buffer"])
    Xunr, yunr = features_5s(ds, idx["test_unrestricted"])

    t0 = time.time()
    clf = HistGradientBoostingClassifier(**HISTGB_FIXED)
    clf.fit(Xtr, ytr)
    pca = clf.predict_proba(Xca)[:, 1]
    thr, cal_f1 = find_best_threshold(yca, pca)
    pbuf = clf.predict_proba(Xbuf)[:, 1]
    punr = clf.predict_proba(Xunr)[:, 1]
    buf_m = metrics_at(ybuf, pbuf, thr)
    unr_m = metrics_at(yunr, punr, thr)

    score_path = os.path.join(SCORES_DIR, f"histgb_fixed_es{event_seed}.npz")
    os.makedirs(SCORES_DIR, exist_ok=True)
    np.savez_compressed(
        score_path,
        probs_buffer=pbuf.astype(np.float32),
        targets_buffer=ybuf.astype(np.float32),
        window_indices_buffer=idx["test_buffer"],
        probs_unrestricted=punr.astype(np.float32),
        targets_unrestricted=yunr.astype(np.float32),
        window_indices_unrestricted=idx["test_unrestricted"],
        thresh=np.asarray([thr], dtype=np.float32),
        used_calib_split=np.asarray([int(use_calib)]),
    )
    return {
        "name": "HistGB_5S_fixedHP",
        "event_seed": event_seed,
        "params": HISTGB_FIXED,
        "n_features": int(Xtr.shape[1]),
        "used_calib_split": use_calib,
        "calib_f1": float(cal_f1),
        "threshold": float(thr),
        "buffer": buf_m,
        "unrestricted": unr_m,
        "score_path": score_path,
        "seconds": round(time.time() - t0, 1),
        "partition_path": part.get("path"),
    }


def phase_histgb(data: Dict, seeds: Sequence[int]) -> Dict:
    log("\n=== Phase HistGB (locked fixed HP, all partitions) ===")
    ds, _, _ = build_scaled_dataset()
    done = {r["event_seed"] for r in data.get("histgb_runs", [])}
    for es in seeds:
        if es in done:
            log(f"  SKIP HistGB es={es}")
            continue
        part = load_partition(es)
        row = run_histgb_partition(es, ds, part)
        data.setdefault("histgb_runs", []).append(row)
        save_results(data)
        b, u = row["buffer"], row["unrestricted"]
        log(
            f"  HistGB es={es}: buf F1={b['f1']:.4f} dF1={b['delta_f1']:+.4f} "
            f"AUROC={b['auroc']:.4f} | unr F1={u['f1']:.4f} AUROC={u['auroc']:.4f} "
            f"pi={u['prevalence']:.3f} ({row['seconds']}s)"
        )
    return data


def phase_deep(
    data: Dict,
    seeds: Sequence[int],
    models: Sequence[str],
    init_seeds: Sequence[int],
    epochs: int,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n=== Phase deep | device={device} | epochs={epochs} | models={list(models)} ===")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
    ds, _, _ = build_scaled_dataset()
    done = {
        deep_key(r["name"], r["event_seed"], r["init_seed"])
        for r in data.get("deep_runs", [])
    }
    jobs = [(m, es, ini) for es in seeds for m in models for ini in init_seeds]
    for i, (name, es, ini) in enumerate(jobs, 1):
        key = deep_key(name, es, ini)
        if key in done:
            log(f"[{i}/{len(jobs)}] SKIP {name} es={es} init={ini}")
            continue
        part = load_partition(es)
        log(f"[{i}/{len(jobs)}] TRAIN {name} es={es} init={ini}")
        row = train_deep_one(name, es, ini, ds, part, device, epochs)
        data.setdefault("deep_runs", []).append(row)
        save_results(data)
        b = row["buffer"]
        log(
            f"    buf F1={b['f1']:.4f} dF1={b['delta_f1']:+.4f} AUROC={b['auroc']:.4f} "
            f"MCC={b['mcc']:.3f} tau={row['threshold']:.2f} "
            f"unr AUROC={row['unrestricted']['auroc']:.4f} ({row['seconds']}s)"
        )
    return data


def write_summary(data: Dict) -> None:
    lines = []
    lines.append("multi-partition SUMMARY (nested select/calib, buffer + unrestricted)")
    lines.append(f"protocol={data.get('meta', {}).get('protocol_version')}")

    # HistGB across partitions
    hrows = data.get("histgb_runs", [])
    if hrows:
        lines.append("\n== HistGB_5S_fixedHP across partitions ==")
        bf1 = [r["buffer"]["f1"] for r in hrows]
        bdf = [r["buffer"]["delta_f1"] for r in hrows]
        bau = [r["buffer"]["auroc"] for r in hrows]
        uau = [r["unrestricted"]["auroc"] for r in hrows]
        lines.append(
            f"n={len(hrows)} buf F1 {np.mean(bf1):.4f}Â±{np.std(bf1, ddof=1) if len(bf1)>1 else 0:.4f} | "
            f"dF1 {np.mean(bdf):+.4f} | AUROC {np.mean(bau):.4f} | "
            f"unr AUROC {np.mean(uau):.4f} | "
            f"above trivial {sum(1 for x in bdf if x>0)}/{len(bdf)} | "
            f"tau_floor {sum(1 for r in hrows if r['buffer']['tau_at_grid_floor'])}/{len(hrows)}"
        )

    # Deep: mean over inits then across partitions
    by = defaultdict(lambda: defaultdict(list))
    for r in data.get("deep_runs", []):
        by[r["name"]][r["event_seed"]].append(r)
    lines.append("\n== Deep models (mean over inits, then across partitions) ==")
    for name, parts in by.items():
        f1s, dfs, aus, uaus, floors = [], [], [], [], []
        for es, runs in parts.items():
            f1s.append(float(np.mean([r["buffer"]["f1"] for r in runs])))
            dfs.append(float(np.mean([r["buffer"]["delta_f1"] for r in runs])))
            aus.append(float(np.mean([r["buffer"]["auroc"] for r in runs])))
            uaus.append(float(np.mean([r["unrestricted"]["auroc"] for r in runs])))
            floors.append(float(np.mean([r["buffer"]["tau_at_grid_floor"] for r in runs])))
        if not f1s:
            continue
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        lines.append(
            f"{name}: n_part={len(f1s)} buf F1 {np.mean(f1s):.4f}Â±{sd(f1s):.4f} | "
            f"dF1 {np.mean(dfs):+.4f}Â±{sd(dfs):.4f} | AUROC {np.mean(aus):.4f}Â±{sd(aus):.4f} | "
            f"unr AUROC {np.mean(uaus):.4f}Â±{sd(uaus):.4f} | "
            f"above trivial {sum(1 for x in dfs if x>0)}/{len(dfs)} | "
            f"mean tau_floor rate {np.mean(floors):.2f}"
        )
        # ranking note
        lines.append(f"  per-partition mean F1: {[round(x,4) for x in f1s]}")

    text = "\n".join(lines) + "\n"
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    log("\n" + text)


def ensure_splits(seeds: Sequence[int]) -> None:
    missing = [es for es in seeds if not os.path.exists(os.path.join(SPLITS_DIR, f"msl_partition_es{es}.npz"))]
    if missing:
        log(f"Freezing missing partitions: {missing}")
        freeze_all(seeds)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "deep", "histgb", "summary", "freeze"], default="all")
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 partition, 1 init, 3 epochs")
    ap.add_argument("--models", nargs="*", default=["CNN-only", "LST-TFDN"])
    ap.add_argument("--include-stdt", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--init-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    if args.freeze_only or args.phase == "freeze":
        freeze_all(args.seeds or EVENT_SEEDS_20)
        return

    seeds = args.seeds or (EVENT_SEEDS_20[:1] if args.smoke else EVENT_SEEDS_20)
    models = list(args.models)
    if args.include_stdt and "Standard Transformer" not in models:
        models.append("Standard Transformer")
    init_seeds = args.init_seeds or (
        INIT_SEEDS_5[:1] if args.smoke else INIT_SEEDS_5
    )
    # StdT uses fewer inits unless overridden
    epochs = 3 if args.smoke else args.epochs

    ensure_splits(seeds)
    data = load_results()
    data["meta"]["torch"] = torch.__version__
    data["meta"]["cuda"] = torch.version.cuda
    data["meta"]["device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    save_results(data)

    if args.phase in ("all", "histgb") and not args.smoke:
        data = phase_histgb(data, seeds)
    elif args.smoke and args.phase in ("all", "histgb"):
        data = phase_histgb(data, seeds)

    if args.phase in ("all", "deep"):
        # For StdT, optionally trim inits inside loop by filtering jobs â€” keep simple:
        deep_inits = init_seeds
        data = phase_deep(data, seeds, models, deep_inits, epochs)

    write_summary(data)
    log(f"Results: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
