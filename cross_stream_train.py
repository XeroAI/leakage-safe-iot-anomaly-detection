"""
cross-stream campaign: PSM + SMAP/MSL per-entity under the same multi-partition protocol.

Overnight profile (default):
  * PSM: 5 event seeds x 3 inits x {CNN-only, LST-TFDN} + HistGB
  * MSL entities (>=3 events): 5 seeds x 3 inits x CNN-only + HistGB
  * SMAP entities (>=3 events): 5 seeds x 1 init x CNN-only + HistGB

Resume-safe. Designed to be started by cross_stream_prepare.py after the multi-partition campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data_loader import TimeSeriesDataset, _anomaly_segments
from model import build_cnn_only_baseline, build_lst_tfdn_baseline
from msl_event_splits import CONTEXT_BUFFER, STRIDE, WINDOW_SIZE, build_partition
from train import find_best_threshold

ROOT = "cross_stream"
SPLITS_DIR = os.path.join(ROOT, "splits")
SCORES_DIR = os.path.join(ROOT, "scores")
RESULTS_JSON = os.path.join(ROOT, "results.json")
RESULTS_TXT = os.path.join(ROOT, "results_summary.txt")
LOG_PATH = os.path.join(ROOT, "runner.log")
MANIFEST = os.path.join(ROOT, "entities_manifest.json")

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
POS_WEIGHT = 10.0
EVENT_SEEDS_5 = [42, 123, 456, 789, 1024]
INIT_SEEDS_3 = [42, 123, 456]
HISTGB_FIXED = dict(
    max_depth=6, learning_rate=0.05, max_iter=300, random_state=42, early_stopping=False
)
BUILDERS = {
    "CNN-only": build_cnn_only_baseline,
    "LST-TFDN": build_lst_tfdn_baseline,
}


def log(msg: str) -> None:
    os.makedirs(ROOT, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
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
        "tau_at_grid_floor": bool(abs(thr - 0.01) < 1e-12),
    }


def features_5s(dataset, indices: np.ndarray):
    xs, ys = [], []
    for wi in indices:
        x, y = dataset[int(wi)]
        x = np.asarray(x, dtype=np.float64)
        t = np.arange(x.shape[1], dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-8)
        xs.append(np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), (x * t[None]).mean(1)]))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def collect_probs(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            p = torch.sigmoid(model(data.to(device))).view(-1).cpu().numpy()
            probs.append(p)
            targets.append(target.numpy())
    if not probs:
        return np.asarray([]), np.asarray([])
    return np.concatenate(probs), np.concatenate(targets)


def load_stream(stream_id: str) -> Tuple[TimeSeriesDataset, np.ndarray]:
    """stream_id: 'PSM' or 'MSL/<chan>' or 'SMAP/<chan>'."""
    if stream_id == "PSM":
        train = np.load("datasets/PSM/PSM_train.npy")
        test = np.load("datasets/PSM/PSM_test.npy")
        labels = np.load("datasets/PSM/PSM_test_label.npy")
    else:
        craft, chan = stream_id.split("/", 1)
        base = os.path.join("datasets", craft, "entities", chan)
        train = np.load(os.path.join(base, "train.npy"))
        test = np.load(os.path.join(base, "test.npy"))
        labels = np.load(os.path.join(base, "test_label.npy"))
    labels = (np.asarray(labels) > 0).astype(np.float32)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    ds = TimeSeriesDataset(test, labels, WINDOW_SIZE, STRIDE)
    return ds, labels.astype(np.int8)


def stream_split_dir(stream_id: str) -> str:
    safe = stream_id.replace("/", "_")
    return os.path.join(SPLITS_DIR, safe)


def freeze_stream_partitions(stream_id: str, labels: np.ndarray, seeds: Sequence[int]) -> List[int]:
    out_dir = stream_split_dir(stream_id)
    os.makedirs(out_dir, exist_ok=True)
    ok = []
    for es in seeds:
        path = os.path.join(out_dir, f"partition_es{es}.npz")
        if os.path.exists(path):
            ok.append(es)
            continue
        part = build_partition(labels, es)
        # require non-empty train and buffer test; calib may be empty for tiny val
        if part["summaries"]["train"]["n_windows"] == 0 or part["summaries"]["test_buffer"]["n_windows"] == 0:
            log(f"  SKIP {stream_id} es={es}: empty train/test")
            continue
        if part["summaries"]["val_select"]["n_windows"] == 0:
            log(f"  SKIP {stream_id} es={es}: empty val_select")
            continue
        np.savez_compressed(
            path,
            meta_json=np.asarray([json.dumps(part["meta"])]),
            events_json=np.asarray([json.dumps(part["events"])]),
            summaries_json=np.asarray([json.dumps(part["summaries"])]),
            train_idx=part["window_indices"]["train"],
            val_select_idx=part["window_indices"]["val_select"],
            val_calib_idx=part["window_indices"]["val_calib"],
            test_buffer_idx=part["window_indices"]["test_buffer"],
            test_unrestricted_idx=part["window_indices"]["test_unrestricted"],
        )
        ok.append(es)
    return ok


def load_part(stream_id: str, event_seed: int) -> Dict:
    path = os.path.join(stream_split_dir(stream_id), f"partition_es{event_seed}.npz")
    z = np.load(path, allow_pickle=True)
    return {
        "meta": json.loads(str(z["meta_json"][0])),
        "events": json.loads(str(z["events_json"][0])),
        "summaries": json.loads(str(z["summaries_json"][0])),
        "window_indices": {
            "train": z["train_idx"].astype(np.int32),
            "val_select": z["val_select_idx"].astype(np.int32),
            "val_calib": z["val_calib_idx"].astype(np.int32),
            "test_buffer": z["test_buffer_idx"].astype(np.int32),
            "test_unrestricted": z["test_unrestricted_idx"].astype(np.int32),
        },
        "path": path,
    }


def load_results() -> Dict:
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {"protocol_version": "phaseC_v1"}, "deep_runs": [], "histgb_runs": []}


def save_results(data: Dict) -> None:
    os.makedirs(ROOT, exist_ok=True)
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, RESULTS_JSON)


def make_loader(ds, indices, shuffle):
    return DataLoader(Subset(ds, indices.tolist()), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_deep_one(name, stream_id, event_seed, init_seed, ds, part, device, epochs):
    idx = part["window_indices"]
    train_loader = make_loader(ds, idx["train"], True)
    sel_loader = make_loader(ds, idx["val_select"], False)
    use_calib = len(idx["val_calib"]) > 0
    cal_loader = make_loader(ds, idx["val_calib"] if use_calib else idx["val_select"], False)
    buf_loader = make_loader(ds, idx["test_buffer"], False)
    unr_loader = make_loader(ds, idx["test_unrestricted"], False)

    num_sensors = int(ds[0][0].shape[0])
    set_seed(init_seed)
    model = BUILDERS[name](num_sensors, WINDOW_SIZE).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_sel, best_state = -1.0, None
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
        sel_p, sel_y = collect_probs(model, sel_loader, device)
        _, sel_f1 = find_best_threshold(sel_y, sel_p)
        if sel_f1 > best_sel:
            best_sel = float(sel_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    cal_p, cal_y = collect_probs(model, cal_loader, device)
    thr, cal_f1 = find_best_threshold(cal_y, cal_p)
    buf_p, buf_y = collect_probs(model, buf_loader, device)
    unr_p, unr_y = collect_probs(model, unr_loader, device)
    safe = stream_id.replace("/", "_")
    score_path = os.path.join(
        SCORES_DIR, f"deep_{safe}_{name.replace(' ', '_')}_es{event_seed}_init{init_seed}.npz"
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
        thresh=np.asarray([thr], dtype=np.float32),
    )
    return {
        "stream_id": stream_id,
        "name": name,
        "event_seed": event_seed,
        "init_seed": init_seed,
        "device": str(device),
        "epochs": epochs,
        "best_select_f1": best_sel,
        "calib_f1": float(cal_f1),
        "used_calib_split": use_calib,
        "threshold": float(thr),
        "buffer": metrics_at(buf_y, buf_p, thr),
        "unrestricted": metrics_at(unr_y, unr_p, thr),
        "score_path": score_path,
        "seconds": round(time.time() - t0, 1),
    }


def run_histgb(stream_id, event_seed, ds, part):
    idx = part["window_indices"]
    Xtr, ytr = features_5s(ds, idx["train"])
    use_calib = len(idx["val_calib"]) > 0
    Xca, yca = features_5s(ds, idx["val_calib"] if use_calib else idx["val_select"])
    Xbuf, ybuf = features_5s(ds, idx["test_buffer"])
    Xunr, yunr = features_5s(ds, idx["test_unrestricted"])
    t0 = time.time()
    clf = HistGradientBoostingClassifier(**HISTGB_FIXED)
    clf.fit(Xtr, ytr)
    thr, cal_f1 = find_best_threshold(yca, clf.predict_proba(Xca)[:, 1])
    pbuf = clf.predict_proba(Xbuf)[:, 1]
    punr = clf.predict_proba(Xunr)[:, 1]
    safe = stream_id.replace("/", "_")
    score_path = os.path.join(SCORES_DIR, f"histgb_{safe}_es{event_seed}.npz")
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
    )
    return {
        "stream_id": stream_id,
        "name": "HistGB_5S_fixedHP",
        "event_seed": event_seed,
        "used_calib_split": use_calib,
        "calib_f1": float(cal_f1),
        "threshold": float(thr),
        "buffer": metrics_at(ybuf, pbuf, thr),
        "unrestricted": metrics_at(yunr, punr, thr),
        "score_path": score_path,
        "seconds": round(time.time() - t0, 1),
    }


def eligible_entities(craft: str, min_events: int = 3) -> List[str]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    out = []
    for row in man.get(craft, []):
        if row["n_events"] >= min_events:
            out.append(f"{craft}/{row['chan_id']}")
    return out


def overnight_jobs() -> List[Dict]:
    """List of {stream_id, seeds, inits, models}."""
    jobs = [
        {
            "stream_id": "PSM",
            "seeds": EVENT_SEEDS_5,
            "inits": INIT_SEEDS_3,
            "models": ["CNN-only", "LST-TFDN"],
        }
    ]
    # Most NASA channels have 1â€“2 labeled events; require >=2 so train/test
    # can be event-exclusive (val may share the train side when tiny).
    for sid in eligible_entities("MSL", 2):
        jobs.append({
            "stream_id": sid,
            "seeds": EVENT_SEEDS_5,
            "inits": INIT_SEEDS_3,
            "models": ["CNN-only"],
        })
    for sid in eligible_entities("SMAP", 2):
        jobs.append({
            "stream_id": sid,
            "seeds": EVENT_SEEDS_5,
            "inits": [42],
            "models": ["CNN-only"],
        })
    return jobs


def run_campaign(jobs: List[Dict], epochs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"cross-stream device={device} epochs={epochs} streams={len(jobs)}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    data = load_results()
    data["meta"]["torch"] = torch.__version__
    data["meta"]["device"] = str(device)
    save_results(data)

    deep_done = {
        (r["stream_id"], r["name"], r["event_seed"], r["init_seed"])
        for r in data.get("deep_runs", [])
    }
    hist_done = {(r["stream_id"], r["event_seed"]) for r in data.get("histgb_runs", [])}

    for job in jobs:
        sid = job["stream_id"]
        try:
            ds, labels = load_stream(sid)
        except Exception as e:
            log(f"SKIP stream {sid}: {e}")
            continue
        n_ev = len(_anomaly_segments(labels))
        log(f"\n=== Stream {sid} | events={n_ev} | sensors={ds[0][0].shape[0]} ===")
        seeds = freeze_stream_partitions(sid, labels, job["seeds"])
        if not seeds:
            log(f"  no valid partitions for {sid}")
            continue

        for es in seeds:
            if (sid, es) in hist_done:
                continue
            part = load_part(sid, es)
            row = run_histgb(sid, es, ds, part)
            data.setdefault("histgb_runs", []).append(row)
            save_results(data)
            b = row["buffer"]
            log(
                f"  HistGB es={es}: buf F1={b['f1']:.4f} dF1={b['delta_f1']:+.4f} "
                f"AUROC={b['auroc']:.4f} ({row['seconds']}s)"
            )
            hist_done.add((sid, es))

        for es in seeds:
            part = load_part(sid, es)
            for name in job["models"]:
                for ini in job["inits"]:
                    key = (sid, name, es, ini)
                    if key in deep_done:
                        continue
                    log(f"  TRAIN {name} es={es} init={ini}")
                    row = train_deep_one(name, sid, es, ini, ds, part, device, epochs)
                    data.setdefault("deep_runs", []).append(row)
                    save_results(data)
                    b = row["buffer"]
                    log(
                        f"    buf F1={b['f1']:.4f} dF1={b['delta_f1']:+.4f} "
                        f"AUROC={b['auroc']:.4f} unrAUROC={row['unrestricted']['auroc']:.4f} "
                        f"({row['seconds']}s)"
                    )
                    deep_done.add(key)

    write_summary(data)


def write_summary(data: Dict) -> None:
    lines = ["cross-stream SUMMARY", f"deep={len(data.get('deep_runs',[]))} histgb={len(data.get('histgb_runs',[]))}"]
    by = {}
    for r in data.get("deep_runs", []):
        by.setdefault(r["stream_id"], {}).setdefault(r["name"], []).append(r["buffer"]["auroc"])
    for sid, models in sorted(by.items()):
        bits = [f"{m} AUROC={np.nanmean(v):.3f} (n={len(v)})" for m, v in models.items()]
        lines.append(f"{sid}: " + " | ".join(bits))
    text = "\n".join(lines) + "\n"
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    log("\n" + text)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["overnight", "psm-only", "smoke"], default="overnight")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    epochs = 3 if args.profile == "smoke" else args.epochs

    if args.profile == "smoke":
        jobs = [{"stream_id": "PSM", "seeds": [42], "inits": [42], "models": ["CNN-only"]}]
    elif args.profile == "psm-only":
        jobs = [{
            "stream_id": "PSM",
            "seeds": EVENT_SEEDS_5,
            "inits": INIT_SEEDS_3,
            "models": ["CNN-only", "LST-TFDN"],
        }]
    else:
        jobs = overnight_jobs()

    run_campaign(jobs, epochs)


if __name__ == "__main__":
    main()
