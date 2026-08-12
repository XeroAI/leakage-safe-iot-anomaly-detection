"""
Like-for-like baseline comparison under the event-level leakage-safe protocol.

Trains LST-TFDN, a ~1M-parameter Standard Transformer, and CNN-only on MSL
with identical data splits, thresholding, and training hyperparameters.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
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
SPLIT_MODE = "event"
DEFAULT_SEEDS = [42, 43, 44]

BASELINES = [
    ("LST-TFDN (Proposed)", build_lst_tfdn_baseline),
    ("Standard Transformer", build_standard_transformer_baseline),
    ("CNN-only", build_cnn_only_baseline),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_ms(model, test_loader, device):
    model.eval()
    times = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            start = time.time()
            model(data)
            times.append(time.time() - start)
    return sum(times) / len(times) * 1000


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs, verbose=True):
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                probs = torch.sigmoid(model(data)).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(target.numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        epoch_thresh, val_f1 = find_best_threshold(val_targets, val_probs)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_thresh = epoch_thresh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"    Epoch {epoch + 1:02d}/{epochs} | Val F1: {val_f1:.4f} | Thresh: {epoch_thresh:.2f}")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_probs, test_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            probs = torch.sigmoid(model(data)).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(target.numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    test_preds = (test_probs > best_thresh).astype(int)

    return {
        "test_f1": float(f1_score(test_targets, test_preds, zero_division=0)),
        "test_precision": float(precision_score(test_targets, test_preds, zero_division=0)),
        "test_recall": float(recall_score(test_targets, test_preds, zero_division=0)),
        "best_val_f1": float(best_val_f1),
        "best_thresh": float(best_thresh),
        "inference_ms": float(measure_inference_ms(model, test_loader, device)),
    }


def run_baselines(epochs=EPOCHS, seeds=None, baselines=None, verbose=True):
    seeds = seeds or DEFAULT_SEEDS
    baselines = baselines or BASELINES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print(
        f"Event-level baseline comparison | {DATASET_NAME} | "
        f"split_mode={SPLIT_MODE} | epochs={epochs} | seeds={seeds} | device={device}"
    )
    print("=" * 100)

    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_NAME, BATCH_SIZE, WINDOW_SIZE, stride=STRIDE, split_mode=SPLIT_MODE
    )
    num_sensors = next(iter(train_loader))[0].shape[1]

    all_runs = []
    for name, build_fn in baselines:
        model = build_fn(num_sensors, WINDOW_SIZE).to(device)
        num_params = count_params(model)
        print(f"\n>>> {name} | {num_params:,} parameters")

        seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            model = build_fn(num_sensors, WINDOW_SIZE).to(device)
            if verbose:
                print(f"  Seed {seed}:")
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device, epochs, verbose=verbose
            )
            metrics.update({"name": name, "seed": seed, "num_params": num_params})
            seed_metrics.append(metrics)
            all_runs.append(metrics)
            print(
                f"    Test -> F1: {metrics['test_f1']:.4f} | "
                f"P: {metrics['test_precision']:.4f} | R: {metrics['test_recall']:.4f} | "
                f"Inf: {metrics['inference_ms']:.1f}ms"
            )

        f1s = np.array([m["test_f1"] for m in seed_metrics])
        print(
            f"  {name} mean F1: {f1s.mean():.4f} +/- {f1s.std(ddof=1) if len(f1s) > 1 else 0.0:.4f}"
        )

    summary = summarize_runs(all_runs, seeds, baselines)
    print_summary_table(summary)
    save_results(all_runs, summary)
    return all_runs, summary


def summarize_runs(all_runs, seeds, baselines=None):
    baselines = baselines or BASELINES
    summary = {"dataset": DATASET_NAME, "split_mode": SPLIT_MODE, "seeds": seeds, "models": []}
    for name, _ in baselines:
        rows = [r for r in all_runs if r["name"] == name]
        if not rows:
            continue
        f1 = np.array([r["test_f1"] for r in rows])
        prec = np.array([r["test_precision"] for r in rows])
        rec = np.array([r["test_recall"] for r in rows])
        summary["models"].append(
            {
                "name": name,
                "num_params": rows[0]["num_params"],
                "inference_ms": float(np.mean([r["inference_ms"] for r in rows])),
                "f1_mean": float(f1.mean()),
                "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "precision_mean": float(prec.mean()),
                "recall_mean": float(rec.mean()),
            }
        )
    return summary


def print_summary_table(summary):
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON SUMMARY (event-level holdout)")
    print("=" * 100)
    print(
        f"{'Model':<28} {'Test F1':>14} {'Precision':>10} {'Recall':>10} "
        f"{'Params':>12} {'Inf (ms)':>10}"
    )
    print("-" * 100)
    for m in summary["models"]:
        f1_str = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
        print(
            f"{m['name']:<28} {f1_str:>14} {m['precision_mean']:>10.4f} {m['recall_mean']:>10.4f} "
            f"{m['num_params']:>12,} {m['inference_ms']:>10.1f}"
        )
    print("=" * 100)


def save_results(all_runs, summary, json_path="baselines_event_results.json", txt_path="baselines_event_results.txt"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs, "summary": summary}, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Event-Level Baseline Comparison — MSL\n")
        f.write("=" * 100 + "\n")
        for m in summary["models"]:
            f.write(
                f"{m['name']}: F1={m['f1_mean']:.4f}±{m['f1_std']:.4f}, "
                f"P={m['precision_mean']:.4f}, R={m['recall_mean']:.4f}, "
                f"Params={m['num_params']:,}, Inf={m['inference_ms']:.1f}ms\n"
            )
        f.write("\nLaTeX table rows:\n")
        for m in summary["models"]:
            f1_cell = f"${m['f1_mean']:.4f} \\pm {m['f1_std']:.4f}$"
            f.write(
                f"{m['name']} & {f1_cell} & {m['precision_mean']:.4f} & {m['recall_mean']:.4f} & "
                f"{m['num_params']:,} & {m['inference_ms']:.1f} \\\\\n"
            )

    print(f"\nSaved {json_path} and {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event-level like-for-like baselines on MSL.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 3 epochs, seed 42, LST-TFDN only.",
    )
    args = parser.parse_args()

    if args.quick:
        run_baselines(epochs=3, seeds=[42], baselines=[BASELINES[0]])
    else:
        run_baselines(epochs=args.epochs, seeds=args.seeds)
