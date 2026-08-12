import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW

from data_loader import get_dataloaders
from model import LST_TFDN


# ==========================================
# Hyperparameters
# ==========================================
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
WINDOW_SIZE = 100
STRIDE = 10
DATASET_NAME = "MSL"
POS_WEIGHT = 10.0
SEED = 42
SPLIT_MODE = "event"  # leakage-safe event-level holdout on official test


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_best_threshold(targets, probs):
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def train_model(
    epochs=EPOCHS,
    max_train_batches=None,
    max_eval_batches=None,
    dataset_name=DATASET_NAME,
    model_save_path=None,
    verbose=True,
    seed=SEED,
    split_mode=SPLIT_MODE,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}", flush=True)
        print(f"Loading {dataset_name} dataset (split_mode={split_mode})...", flush=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, BATCH_SIZE, WINDOW_SIZE, stride=STRIDE, split_mode=split_mode
    )

    sample_data, _ = next(iter(train_loader))
    num_sensors = sample_data.shape[1]
    if verbose:
        print(f"Number of sensors detected: {num_sensors}", flush=True)

    model = LST_TFDN(
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        d_model=32,
        heads=4,
        dropout=0.2,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total Trainable Parameters: {total_params:,}", flush=True)

    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if verbose:
        print(f"Starting training for {epochs} epochs...", flush=True)
        print("-" * 90, flush=True)

    best_val_f1 = 0.0
    best_thresh = 0.5
    best_model_path = model_save_path or f"lst_tfdn_{dataset_name}_best.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        train_loss /= train_batches
        epoch_time = time.time() - start_time

        model.eval()
        val_probs = []
        val_targets = []
        inference_times = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader, start=1):
                data = data.to(device)

                start_inf = time.time()
                logits = model(data)
                inference_times.append(time.time() - start_inf)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(target.numpy())

                if max_eval_batches is not None and batch_idx >= max_eval_batches:
                    break

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)

        epoch_thresh, val_f1 = find_best_threshold(val_targets, val_probs)
        val_preds = (val_probs > epoch_thresh).astype(int)
        precision = precision_score(val_targets, val_preds, zero_division=0)
        recall = recall_score(val_targets, val_preds, zero_division=0)
        avg_inf_time = sum(inference_times) / len(inference_times) * 1000
        pred_pos = int(val_preds.sum())
        target_pos = int(val_targets.sum())

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_thresh = epoch_thresh
            torch.save(model.state_dict(), best_model_path)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | Time: {epoch_time:.1f}s | "
                f"Loss: {train_loss:.4f} | Thresh: {epoch_thresh:.2f} | "
                f"Val Prec: {precision:.4f} | Val Rec: {recall:.4f} | Val F1: {val_f1:.4f} | "
                f"Inf: {avg_inf_time:.1f}ms | Pred+: {pred_pos} / Target+: {target_pos}",
                flush=True,
            )

    if verbose:
        print("-" * 90, flush=True)
        print("Training complete! Loading best model for final test evaluation...", flush=True)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader, start=1):
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(target.numpy())

            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    test_preds = (test_probs > best_thresh).astype(int)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    if verbose:
        print(
            f"Final Test (thresh={best_thresh:.2f}) -> "
            f"Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}",
            flush=True,
        )

    return {
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "best_val_f1": best_val_f1,
        "best_thresh": best_thresh,
        "num_params": total_params,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LST-TFDN on IoT anomaly data.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
