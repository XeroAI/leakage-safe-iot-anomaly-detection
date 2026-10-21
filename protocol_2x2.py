"""
Population-pool 2x2 protocol comparison on MSL.

Factors
  leakage:   event-exclusive  |  random-stratified
  population: buffer-retained |  unrestricted (full held-out territories)

Matched cell: random-stratified split drawn from the SAME window pool as the
event-exclusive buffer train+val+test for that partition (same prevalence /
population; only assignment differs).

Event-exclusive cells reuse multi-partition frozen scores (no retrain).
Stratified cells are trained here (CNN-only, LST-TFDN, HistGB_5S_fixedHP).

Usage:
  python protocol_2x2.py --smoke
  python protocol_2x2.py --seeds 42 123 456 789 1024 --inits 42 123 456
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from model import build_cnn_only_baseline, build_lst_tfdn_baseline
from msl_event_splits import (
    EVENT_SEEDS_20,
    WINDOW_SIZE,
    STRIDE,
    build_scaled_dataset,
    load_partition,
)
from train import find_best_threshold

ROOT = "protocol_2x2"
SCORES = os.path.join(ROOT, "scores")
RESULTS = os.path.join(ROOT, "results.json")
SUMMARY = os.path.join(ROOT, "results_summary.txt")
LOG = os.path.join(ROOT, "runner.log")
MULTIPART_SCORES = os.path.join("msl_multipartition", "scores")

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
POS_WEIGHT = 10.0
HISTGB = dict(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42, early_stopping=False)
BUILDERS = {"CNN-only": build_cnn_only_baseline, "LST-TFDN": build_lst_tfdn_baseline}


def log(msg: str) -> None:
    os.makedirs(ROOT, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_at(y, p, thr) -> Dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > thr).astype(int)
    pi = float(y.mean()) if len(y) else float("nan")
    trivial = 2 * pi / (1 + pi) if pi > 0 else 0.0
    f1 = float(f1_score(y, pred, zero_division=0))
    auroc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    auprc = float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")
    return {
        "f1": f1,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_norm": auprc / pi if pi > 0 else float("nan"),
        "prevalence": pi,
        "trivial_f1": trivial,
        "delta_f1": f1 - trivial,
        "thresh": float(thr),
        "n": int(len(y)),
    }


def features_5s(ds, indices):
    xs, ys = [], []
    for wi in indices:
        x, y = ds[int(wi)]
        x = np.asarray(x, dtype=np.float64)
        t = np.arange(x.shape[1], dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-8)
        xs.append(np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), (x * t[None]).mean(1)]))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def collect_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for data, target in loader:
            ps.append(torch.sigmoid(model(data.to(device))).view(-1).cpu().numpy())
            ys.append(target.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def stratified_split(indices: np.ndarray, labels: np.ndarray, seed: int, train_ratio=0.5):
    """50/25/25 stratified on window labels; returns train, val, test index arrays (into `indices`)."""
    y = labels
    if len(np.unique(y)) < 2 or len(indices) < 6:
        # fallback: random permutation ratios
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(indices))
        n_tr = max(1, int(round(len(indices) * train_ratio)))
        n_va = max(1, int(round(len(indices) * 0.25)))
        tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
        if len(te) == 0:
            te = va[-1:]
            va = va[:-1]
        return indices[tr], indices[va], indices[te]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=seed)
    tr_rel, temp_rel = next(sss1.split(np.zeros(len(y)), y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    va_rel, te_rel = next(sss2.split(np.zeros(len(temp_rel)), y[temp_rel]))
    return indices[tr_rel], indices[temp_rel[va_rel]], indices[temp_rel[te_rel]]


def window_labels(ds, indices):
    return np.asarray([int(ds[int(i)][1].item()) for i in indices], dtype=int)


def make_loader(ds, indices, shuffle):
    return DataLoader(Subset(ds, indices.tolist()), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_deep(name, ds, tr, va, te, device, epochs, init_seed):
    set_seed(init_seed)
    # nest: first half of va for select, second for calib (by order)
    if len(va) >= 4:
        mid = len(va) // 2
        sel, cal = va[:mid], va[mid:]
    else:
        sel = cal = va
    train_loader = make_loader(ds, tr, True)
    sel_loader = make_loader(ds, sel, False)
    cal_loader = make_loader(ds, cal, False)
    te_loader = make_loader(ds, te, False)
    model = BUILDERS[name](int(ds[0][0].shape[0]), WINDOW_SIZE).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_f, best_state = -1.0, None
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = crit(model(data).view(-1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sp, sy = collect_probs(model, sel_loader, device)
        _, sf = find_best_threshold(sy, sp)
        if sf > best_f:
            best_f = float(sf)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    cp, cy = collect_probs(model, cal_loader, device)
    thr, _ = find_best_threshold(cy, cp)
    tp, ty = collect_probs(model, te_loader, device)
    return metrics_at(ty, tp, thr), {"seconds": round(time.time() - t0, 1), "thresh": thr,
                                      "probs": tp, "targets": ty, "test_idx": te}


def train_histgb(ds, tr, va, te):
    Xtr, ytr = features_5s(ds, tr)
    Xva, yva = features_5s(ds, va)
    Xte, yte = features_5s(ds, te)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(**HISTGB)
    clf.fit(Xtr, ytr)
    thr, _ = find_best_threshold(yva, clf.predict_proba(Xva)[:, 1])
    pte = clf.predict_proba(Xte)[:, 1]
    return metrics_at(yte, pte, thr), {"seconds": round(time.time() - t0, 1), "thresh": thr,
                                       "probs": pte, "targets": yte, "test_idx": te}


def load_event_cell(name: str, event_seed: int, init_seed: int, population: str) -> Dict:
    """Reuse multi-partition scores for event-exclusive cells."""
    if name == "HistGB_5S_fixedHP":
        path = os.path.join(MULTIPART_SCORES, f"histgb_fixed_es{event_seed}.npz")
        z = np.load(path)
        thr = float(z["thresh"][0])
        if population == "buffer":
            return metrics_at(z["targets_buffer"], z["probs_buffer"], thr)
        return metrics_at(z["targets_unrestricted"], z["probs_unrestricted"], thr)
    path = os.path.join(
        MULTIPART_SCORES,
        f"deep_{name.replace(' ', '_')}_es{event_seed}_init{init_seed}.npz",
    )
    z = np.load(path)
    thr = float(z["thresh"][0])
    if population == "buffer":
        return metrics_at(z["targets_buffer"], z["probs_buffer"], thr)
    return metrics_at(z["targets_unrestricted"], z["probs_unrestricted"], thr)


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {"protocol": "m4_2x2_v1"}, "cells": []}


def save_results(data):
    os.makedirs(ROOT, exist_ok=True)
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, RESULTS)


def cell_key(row):
    return (row["leakage"], row["population"], row["model"], row["event_seed"], row.get("init_seed", 0))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--inits", type=int, nargs="*", default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", nargs="*", default=["CNN-only", "LST-TFDN", "HistGB_5S_fixedHP"])
    args = ap.parse_args()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    inits = args.inits[:1] if args.smoke else args.inits
    epochs = 3 if args.smoke else args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Protocol 2x2 | device={device} | seeds={seeds} | inits={inits} | epochs={epochs}")

    ds, labels, _ = build_scaled_dataset()
    n_windows = len(ds)
    all_idx = np.arange(n_windows, dtype=np.int32)
    all_y = window_labels(ds, all_idx)

    data = load_results()
    done = {cell_key(r) for r in data.get("cells", [])}

    for es in seeds:
        part = load_partition(es)
        idx = part["window_indices"]
        pool = np.unique(np.concatenate([
            idx["train"], idx["val_select"], idx["val_calib"], idx["test_buffer"]
        ])).astype(np.int32)
        pool_y = window_labels(ds, pool)
        unr_pool = idx["test_unrestricted"]  # evaluation population only for event-unr
        log(
            f"\n=== event_seed={es} | buffer pool n={len(pool)} pi={pool_y.mean():.3f} "
            f"| full n={n_windows} pi={all_y.mean():.3f} | unr test n={len(unr_pool)} ==="
        )

        # --- Event-exclusive cells (reuse multi-partition) ---
        for name in args.models:
            init_list = [0] if name.startswith("HistGB") else inits
            for ini in init_list:
                for pop in ("buffer", "unrestricted"):
                    row = {
                        "leakage": "event-exclusive",
                        "population": pop,
                        "model": name,
                        "event_seed": es,
                        "init_seed": ini,
                        "source": "phaseB_scores",
                    }
                    if cell_key(row) in done:
                        continue
                    try:
                        row["metrics"] = load_event_cell(name, es, ini, pop)
                    except FileNotFoundError as e:
                        log(f"  missing multi-partition score for {name} es={es} init={ini}: {e}")
                        continue
                    data["cells"].append(row)
                    done.add(cell_key(row))
                    save_results(data)
                    m = row["metrics"]
                    log(
                        f"  [event|{pop:13s}|{name:16s}|init={ini}] "
                        f"F1={m['f1']:.4f} dF1={m['delta_f1']:+.4f} AUROC={m['auroc']:.4f} pi={m['prevalence']:.3f}"
                    )

        # --- Stratified-matched: same buffer pool (includes train-region normals; lower pi) ---
        for split_seed in inits:  # use init list as stratified realization seeds
            tr, va, te = stratified_split(pool, pool_y, seed=1000 * es + split_seed)
            # store pool meta once per realization via first model
            for name in args.models:
                row = {
                    "leakage": "random-stratified",
                    "population": "buffer-matched",
                    "model": name,
                    "event_seed": es,
                    "init_seed": split_seed,
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                    "pool_n": int(len(pool)),
                    "pool_prevalence": float(pool_y.mean()),
                }
                if cell_key(row) in done:
                    continue
                log(f"  TRAIN stratified-matched {name} es={es} split={split_seed}")
                if name.startswith("HistGB"):
                    met, extra = train_histgb(ds, tr, va, te)
                else:
                    met, extra = train_deep(name, ds, tr, va, te, device, epochs, split_seed)
                row["metrics"] = met
                row["seconds"] = extra["seconds"]
                # save scores
                os.makedirs(SCORES, exist_ok=True)
                sp = os.path.join(
                    SCORES,
                    f"strat_matched_{name.replace(' ', '_')}_es{es}_init{split_seed}.npz",
                )
                np.savez_compressed(
                    sp, probs=extra["probs"].astype(np.float32),
                    targets=extra["targets"].astype(np.float32),
                    test_idx=extra["test_idx"].astype(np.int32),
                    thresh=np.asarray([extra["thresh"]], np.float32),
                )
                row["score_path"] = sp
                data["cells"].append(row)
                done.add(cell_key(row))
                save_results(data)
                log(
                    f"    F1={met['f1']:.4f} dF1={met['delta_f1']:+.4f} AUROC={met['auroc']:.4f} "
                    f"pi={met['prevalence']:.3f} ({extra['seconds']}s)"
                )

        # --- Prevalence+population matched: valâˆªtest buffer only (excludes train normals) ---
        # Matches event-exclusive test prevalence (~0.48) and near-event buffer population.
        vt_pool = np.unique(
            np.concatenate([idx["val_select"], idx["val_calib"], idx["test_buffer"]])
        ).astype(np.int32)
        vt_y = window_labels(ds, vt_pool)
        log(
            f"  vt-matched pool n={len(vt_pool)} pi={vt_y.mean():.3f} "
            f"(event test_buffer pi={window_labels(ds, idx['test_buffer']).mean():.3f})"
        )
        for split_seed in inits:
            tr, va, te = stratified_split(vt_pool, vt_y, seed=3000 * es + split_seed)
            for name in args.models:
                row = {
                    "leakage": "random-stratified",
                    "population": "buffer-vt-matched",
                    "model": name,
                    "event_seed": es,
                    "init_seed": split_seed,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "n_test": int(len(te)),
                    "pool_n": int(len(vt_pool)),
                    "pool_prevalence": float(vt_y.mean()),
                }
                if cell_key(row) in done:
                    continue
                log(f"  TRAIN stratified-vt-matched {name} es={es} split={split_seed}")
                if name.startswith("HistGB"):
                    met, extra = train_histgb(ds, tr, va, te)
                else:
                    met, extra = train_deep(name, ds, tr, va, te, device, epochs, split_seed)
                row["metrics"] = met
                row["seconds"] = extra["seconds"]
                os.makedirs(SCORES, exist_ok=True)
                sp = os.path.join(
                    SCORES,
                    f"strat_vt_matched_{name.replace(' ', '_')}_es{es}_init{split_seed}.npz",
                )
                np.savez_compressed(
                    sp,
                    probs=extra["probs"].astype(np.float32),
                    targets=extra["targets"].astype(np.float32),
                    test_idx=extra["test_idx"].astype(np.int32),
                    thresh=np.asarray([extra["thresh"]], np.float32),
                )
                row["score_path"] = sp
                data["cells"].append(row)
                done.add(cell_key(row))
                save_results(data)
                log(
                    f"    F1={met['f1']:.4f} dF1={met['delta_f1']:+.4f} AUROC={met['auroc']:.4f} "
                    f"pi={met['prevalence']:.3f} ({extra['seconds']}s)"
                )

        # --- Stratified-full: all windows on official test series ---
        for split_seed in inits:
            tr, va, te = stratified_split(all_idx, all_y, seed=2000 * es + split_seed)
            for name in args.models:
                row = {
                    "leakage": "random-stratified",
                    "population": "full-stream",
                    "model": name,
                    "event_seed": es,  # paired with same event partition seed for bookkeeping
                    "init_seed": split_seed,
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                    "pool_n": int(len(all_idx)),
                    "pool_prevalence": float(all_y.mean()),
                }
                if cell_key(row) in done:
                    continue
                log(f"  TRAIN stratified-full {name} es={es} split={split_seed}")
                if name.startswith("HistGB"):
                    met, extra = train_histgb(ds, tr, va, te)
                else:
                    met, extra = train_deep(name, ds, tr, va, te, device, epochs, split_seed)
                row["metrics"] = met
                row["seconds"] = extra["seconds"]
                sp = os.path.join(
                    SCORES,
                    f"strat_full_{name.replace(' ', '_')}_es{es}_init{split_seed}.npz",
                )
                np.savez_compressed(
                    sp, probs=extra["probs"].astype(np.float32),
                    targets=extra["targets"].astype(np.float32),
                    test_idx=extra["test_idx"].astype(np.int32),
                    thresh=np.asarray([extra["thresh"]], np.float32),
                )
                row["score_path"] = sp
                data["cells"].append(row)
                done.add(cell_key(row))
                save_results(data)
                log(
                    f"    F1={met['f1']:.4f} dF1={met['delta_f1']:+.4f} AUROC={met['auroc']:.4f} "
                    f"pi={met['prevalence']:.3f} ({extra['seconds']}s)"
                )

    write_summary(data)
    log(f"Done. Results: {RESULTS}")


def write_summary(data: Dict) -> None:
    # aggregate mean over inits per (leakage, population, model)
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in data.get("cells", []):
        key = (r["leakage"], r["population"], r["model"])
        buckets[key].append(r["metrics"])
    lines = ["Protocol 2x2 SUMMARY (mean over realizations)", ""]
    lines.append(f"{'leakage':20s} {'population':16s} {'model':18s} {'F1':>7s} {'dF1':>8s} {'AUROC':>7s} {'pi':>6s} n")
    for key in sorted(buckets):
        ms = buckets[key]
        f1 = np.mean([m["f1"] for m in ms])
        df = np.mean([m["delta_f1"] for m in ms])
        au = np.nanmean([m["auroc"] for m in ms])
        pi = np.mean([m["prevalence"] for m in ms])
        lines.append(
            f"{key[0]:20s} {key[1]:16s} {key[2]:18s} {f1:7.4f} {df:+8.4f} {au:7.4f} {pi:6.3f} {len(ms)}"
        )
    # matched contrasts vs event-buffer
    lines.append("")
    lines.append("== Contrast (random-stratified buffer-matched âˆ’ event-exclusive buffer) ==")
    lines.append("   (full buffer pool incl. train normals; prevalence typically lower)")
    for model in ["CNN-only", "LST-TFDN", "HistGB_5S_fixedHP"]:
        a = buckets.get(("random-stratified", "buffer-matched", model), [])
        b = buckets.get(("event-exclusive", "buffer", model), [])
        if a and b:
            lines.append(
                f"{model:18s} Î”F1={np.mean([m['f1'] for m in a]) - np.mean([m['f1'] for m in b]):+.4f} | "
                f"Î”AUROC={np.nanmean([m['auroc'] for m in a]) - np.nanmean([m['auroc'] for m in b]):+.4f} | "
                f"pi_matched={np.mean([m['prevalence'] for m in a]):.3f} "
                f"pi_event={np.mean([m['prevalence'] for m in b]):.3f}"
            )
    lines.append("")
    lines.append(
        "== Prevalence+population matched "
        "(random-stratified buffer-vt-matched âˆ’ event-exclusive buffer) =="
    )
    lines.append("   (valâˆªtest buffer only; prevalence aligned with event test)")
    for model in ["CNN-only", "LST-TFDN", "HistGB_5S_fixedHP"]:
        a = buckets.get(("random-stratified", "buffer-vt-matched", model), [])
        b = buckets.get(("event-exclusive", "buffer", model), [])
        if a and b:
            lines.append(
                f"{model:18s} Î”F1={np.mean([m['f1'] for m in a]) - np.mean([m['f1'] for m in b]):+.4f} | "
                f"Î”AUROC={np.nanmean([m['auroc'] for m in a]) - np.nanmean([m['auroc'] for m in b]):+.4f} | "
                f"pi_vt={np.mean([m['prevalence'] for m in a]):.3f} "
                f"pi_event={np.mean([m['prevalence'] for m in b]):.3f}"
            )
    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text)
    log("\n" + text)


if __name__ == "__main__":
    main()
