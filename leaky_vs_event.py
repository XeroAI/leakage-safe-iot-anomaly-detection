"""
Tier B: leaky stratified vs event-level protocol comparison on MSL.

Trains the same models under both split modes and reports mean F1 + rankings.
Resume-safe: completed (protocol, model, seed) keys are skipped.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from torch.optim import AdamW

from data_loader import get_dataloaders
from model import (
    build_cnn_only_baseline,
    build_lst_tfdn_baseline,
    build_standard_transformer_baseline,
)
from train import find_best_threshold

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
WINDOW_SIZE = 100
STRIDE = 10
POS_WEIGHT = 10.0
DATASET_NAME = "MSL"
EVENT_SEED = 42
SEEDS = [42, 123, 456]  # three seeds for ranking evidence (resume-safe to extend)
RESULTS_JSON = "tierB_leaky_vs_event_results.json"
RESULTS_TXT = "tierB_leaky_vs_event_results.txt"

PROTOCOLS = [
    ("event", "Event-level (leakage-safe)"),
    ("stratified", "Stratified windows (leaky)"),
]

MODELS = [
    ("LST-TFDN", build_lst_tfdn_baseline),
    ("Standard Transformer", build_standard_transformer_baseline),
    ("CNN-only", build_cnn_only_baseline),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auroc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true, y_score) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def collect_probs(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            probs.extend(torch.sigmoid(model(data)).cpu().numpy())
            targets.extend(target.numpy())
    return np.asarray(probs, dtype=float), np.asarray(targets, dtype=float)


def train_one(model, train_loader, val_loader, test_loader, device, epochs: int) -> Dict:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
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
            best_val_f1 = float(val_f1)
            best_thresh = float(epoch_thresh)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
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
        "n_train": int(len(train_loader.dataset)),
        "n_val": int(len(val_loader.dataset)),
        "n_test": int(len(test_loader.dataset)),
        "test_pos_rate": float(np.mean(test_targets)),
    }


def run_key(protocol: str, name: str, seed: int) -> str:
    return f"{protocol}||{name}||seed={seed}"


def load_results() -> Dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": {}, "meta": {}}


def save_results(blob: Dict) -> None:
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def summarize(blob: Dict) -> str:
    lines: List[str] = []
    lines.append("Tier B — Leaky stratified vs event-level (MSL)")
    lines.append(f"Seeds={SEEDS} | epochs={EPOCHS} | w={POS_WEIGHT} | event_seed={EVENT_SEED}")
    lines.append("")

    for protocol, proto_label in PROTOCOLS:
        lines.append(f"=== {proto_label} ({protocol}) ===")
        ranking: List[Tuple[float, str, float]] = []
        for name, _ in MODELS:
            f1s = []
            aurocs = []
            for seed in SEEDS:
                k = run_key(protocol, name, seed)
                if k not in blob["runs"]:
                    continue
                f1s.append(blob["runs"][k]["test_f1"])
                aurocs.append(blob["runs"][k]["test_auroc"])
            if not f1s:
                lines.append(f"  {name}: incomplete")
                continue
            mean_f1 = float(np.mean(f1s))
            std_f1 = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
            mean_auroc = float(np.nanmean(aurocs))
            ranking.append((mean_f1, name, mean_auroc))
            lines.append(
                f"  {name}: F1 {mean_f1:.4f}±{std_f1:.4f} | AUROC {mean_auroc:.4f} | n={len(f1s)}"
            )
        ranking.sort(reverse=True)
        lines.append("  Ranking by mean F1: " + " > ".join(n for _, n, _ in ranking))
        lines.append("")

    # Cross-protocol deltas for each model
    lines.append("=== Event − Stratified ΔF1 (mean) ===")
    for name, _ in MODELS:
        ev, st = [], []
        for seed in SEEDS:
            ke, ks = run_key("event", name, seed), run_key("stratified", name, seed)
            if ke in blob["runs"] and ks in blob["runs"]:
                ev.append(blob["runs"][ke]["test_f1"])
                st.append(blob["runs"][ks]["test_f1"])
        if ev:
            lines.append(f"  {name}: ΔF1={float(np.mean(ev)-np.mean(st)):+.4f} "
                         f"(event {float(np.mean(ev)):.4f} vs leaky {float(np.mean(st)):.4f})")
    return "\n".join(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    blob = load_results()
    blob["meta"] = {
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "pos_weight": POS_WEIGHT,
        "event_seed": EVENT_SEED,
        "window": WINDOW_SIZE,
        "stride": STRIDE,
        "started": blob.get("meta", {}).get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
    }

    # Cache loaders per protocol (same for all models/seeds of that protocol)
    loaders = {}
    for protocol, proto_label in PROTOCOLS:
        print(f"\nBuilding loaders: {proto_label}")
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=DATASET_NAME,
            batch_size=BATCH_SIZE,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            split_mode=protocol,
            event_seed=EVENT_SEED,
        )
        loaders[protocol] = (train_loader, val_loader, test_loader)
        # one batch to get num_sensors
        sample, _ = next(iter(train_loader))
        num_sensors = int(sample.shape[1])
        print(
            f"  sensors={num_sensors} | "
            f"n_train={len(train_loader.dataset)} | "
            f"n_val={len(val_loader.dataset)} | "
            f"n_test={len(test_loader.dataset)}"
        )

    sample, _ = next(iter(loaders["event"][0]))
    num_sensors = int(sample.shape[1])

    for protocol, proto_label in PROTOCOLS:
        train_loader, val_loader, test_loader = loaders[protocol]
        for name, builder in MODELS:
            for seed in SEEDS:
                key = run_key(protocol, name, seed)
                if key in blob["runs"]:
                    print(f"[skip] {key}")
                    continue
                print(f"\n>>> {proto_label} | {name} | seed={seed}")
                set_seed(seed)
                model = builder(num_sensors, WINDOW_SIZE).to(device)
                t0 = time.time()
                metrics = train_one(
                    model, train_loader, val_loader, test_loader, device, EPOCHS
                )
                metrics["elapsed_sec"] = float(time.time() - t0)
                metrics["params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
                blob["runs"][key] = metrics
                save_results(blob)
                print(
                    f"    Test F1={metrics['test_f1']:.4f} | "
                    f"AUROC={metrics['test_auroc']:.4f} | "
                    f"{metrics['elapsed_sec']:.0f}s"
                )

    summary = summarize(blob)
    print("\n" + summary)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSaved {RESULTS_JSON} and {RESULTS_TXT}")


if __name__ == "__main__":
    main()
