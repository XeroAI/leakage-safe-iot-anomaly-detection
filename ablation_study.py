"""
LST-TFDN ablation study on MSL under the event-level leakage-safe protocol.

Matches main baseline settings: 50 epochs, seeds [42, 43, 44], same split/thresholding.
"""

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW

from data_loader import get_dataloaders
from model import LST_TFDN
from train import find_best_threshold


EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
WINDOW_SIZE = 100
STRIDE = 10
D_MODEL = 32
POS_WEIGHT = 10.0
DATASET_NAME = "MSL"
SPLIT_MODE = "event"
DEFAULT_SEEDS = [42, 43, 44]
RESULTS_JSON = "ablation_event_results.json"
RESULTS_TXT = "ablation_results.txt"

ABLATION_CONFIGS = [
    {
        "name": "Full LST-TFDN (Proposed)",
        "use_linear_attn": True,
        "use_fe": True,
        "use_cnn": True,
        "use_transformer": True,
    },
    {
        "name": "CNN-only (no Transformer)",
        "use_linear_attn": True,
        "use_fe": True,
        "use_cnn": True,
        "use_transformer": False,
    },
    {
        "name": "w/ Standard Attention",
        "use_linear_attn": False,
        "use_fe": True,
        "use_cnn": True,
        "use_transformer": True,
    },
    {
        "name": "w/ Standard PE (No FE)",
        "use_linear_attn": True,
        "use_fe": False,
        "use_cnn": True,
        "use_transformer": True,
    },
    {
        "name": "w/o 1D CNN Front-end",
        "use_linear_attn": True,
        "use_fe": True,
        "use_cnn": False,
        "use_transformer": True,
    },
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


def train_variant(config, train_loader, val_loader, test_loader, num_sensors, device, epochs):
    model = LST_TFDN(
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        d_model=D_MODEL,
        heads=4,
        dropout=0.2,
        use_linear_attn=config["use_linear_attn"],
        use_fe=config["use_fe"],
        use_cnn=config["use_cnn"],
        use_transformer=config.get("use_transformer", True),
    ).to(device)

    num_params = count_params(model)
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_thresh = 0.5
    best_state = None

    print(f"\nTraining: {config['name']} ({epochs} epochs, {num_params:,} params)")

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:02d}/{epochs} | Val F1: {val_f1:.4f}")

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

    avg_inf_ms = measure_inference_ms(model, test_loader, device)

    return {
        "name": config["name"],
        "test_f1": float(f1_score(test_targets, test_preds, zero_division=0)),
        "test_precision": float(precision_score(test_targets, test_preds, zero_division=0)),
        "test_recall": float(recall_score(test_targets, test_preds, zero_division=0)),
        "num_params": int(num_params),
        "inference_ms": float(avg_inf_ms),
        "best_thresh": float(best_thresh),
        "best_val_f1": float(best_val_f1),
    }


def load_checkpoint(path=RESULTS_JSON):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runs", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_checkpoint(runs, path=RESULTS_JSON):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"protocol": "event", "epochs": EPOCHS, "runs": runs}, f, indent=2)


def aggregate(runs):
    by_name = defaultdict(list)
    for r in runs:
        by_name[r["name"]].append(r)

    summary = []
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        items = by_name.get(name, [])
        if not items:
            continue
        f1s = [x["test_f1"] for x in items]
        precs = [x["test_precision"] for x in items]
        recs = [x["test_recall"] for x in items]
        infs = [x["inference_ms"] for x in items]
        summary.append({
            "name": name,
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "test_precision": float(np.mean(precs)),
            "test_recall": float(np.mean(recs)),
            "num_params": items[0]["num_params"],
            "inference_ms": float(np.mean(infs)),
            "n_seeds": len(items),
        })
    return summary


def save_results_txt(summary, path=RESULTS_TXT):
    with open(path, "w", encoding="utf-8") as f:
        f.write("LST-TFDN Ablation Study — MSL (event-level holdout)\n")
        f.write("=" * 100 + "\n")
        f.write(
            f"{'Variant':<32} {'Test F1':>18} {'Precision':>10} {'Recall':>10} "
            f"{'Params':>12} {'Inf (ms)':>10}\n"
        )
        f.write("-" * 100 + "\n")
        for r in summary:
            f1_str = f"{r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f}"
            f.write(
                f"{r['name']:<32} {f1_str:>18} "
                f"{r['test_precision']:>10.4f} {r['test_recall']:>10.4f} "
                f"{r['num_params']:>12,} {r['inference_ms']:>10.1f}\n"
            )
        f.write("\nLaTeX table rows:\n")
        for r in summary:
            f.write(
                f"{r['name']} & ${r['test_f1_mean']:.4f} \\pm {r['test_f1_std']:.4f}$ & "
                f"{r['num_params']:,} & {r['inference_ms']:.1f} \\\\\n"
            )
    print(f"\nResults saved to {path}")


def run_ablation(epochs=EPOCHS, seeds=None, configs=None, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or DEFAULT_SEEDS
    configs = configs or ABLATION_CONFIGS

    print(
        f"Ablation study on {DATASET_NAME} | {SPLIT_MODE} | device: {device} | "
        f"{epochs} epochs | seeds={seeds}"
    )
    print("=" * 100)

    runs = load_checkpoint() if resume else []
    done = {(r["name"], r["seed"]) for r in runs}

    for seed in seeds:
        for config in configs:
            key = (config["name"], seed)
            if key in done:
                print(f"Skip (done): {config['name']} seed={seed}")
                continue

            set_seed(seed)
            train_loader, val_loader, test_loader = get_dataloaders(
                DATASET_NAME, BATCH_SIZE, WINDOW_SIZE, stride=STRIDE, split_mode=SPLIT_MODE
            )
            num_sensors = next(iter(train_loader))[0].shape[1]

            print(f"\n>>> {config['name']} | seed={seed}")
            result = train_variant(
                config, train_loader, val_loader, test_loader, num_sensors, device, epochs
            )
            result["seed"] = seed
            runs.append(result)
            save_checkpoint(runs)
            done.add(key)
            print(
                f"  -> Test F1: {result['test_f1']:.4f} | "
                f"Params: {result['num_params']:,} | "
                f"Inference: {result['inference_ms']:.1f}ms"
            )

    summary = aggregate(runs)
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY (MSL, event-level, mean±std over seeds)")
    print("=" * 100)
    for r in summary:
        print(
            f"{r['name']:<32} F1={r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} "
            f"| P={r['test_precision']:.4f} | R={r['test_recall']:.4f} "
            f"| params={r['num_params']:,} | inf={r['inference_ms']:.1f}ms "
            f"| n={r['n_seeds']}"
        )

    save_results_txt(summary)
    save_checkpoint(runs)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LST-TFDN event-level ablation on MSL.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: Full model only, 3 epochs, 1 seed.",
    )
    args = parser.parse_args()

    if args.quick:
        run_ablation(epochs=3, seeds=[42], configs=[ABLATION_CONFIGS[0]], resume=False)
    else:
        run_ablation(epochs=args.epochs, seeds=args.seeds, resume=not args.no_resume)
