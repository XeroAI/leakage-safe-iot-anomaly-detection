"""
Freeze duration-balanced MSL event partitions for multi-partition evaluation.

Unified assignment rule: duration-balanced event assignment with
seeded_variation=True (distinct event_seed => distinct partition).

For each partition we freeze:
  * event IDs in train / val_select / val_calib / test
  * buffer-defined window indices (B=200) for those four roles
  * unrestricted-stream test window indices (midpoint territories)
  * event counts, anomalous durations, prevalences

Nothing is trained here. Runner loads these freezes so every later table
shares one versioned split definition.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_loader import (
    TimeSeriesDataset,
    _anomaly_segments,
    _assign_segments_by_duration,
    load_dataset,
)

WINDOW_SIZE = 100
STRIDE = 10
CONTEXT_BUFFER = 200
SPLITS_DIR = os.path.join("msl_multipartition", "splits")

# 20 fixed event seeds (unified seeded assignment). Do not silently reorder.
EVENT_SEEDS_20: List[int] = [
    42, 123, 456, 789, 1024,
    7, 13, 21, 33, 55,
    89, 144, 233, 377, 610,
    1009, 2026, 3141, 4096, 8191,
]


def _split_val_events(
    val_events: Set[int],
    segments: List[Tuple[int, int]],
    nested_seed: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Split validation events into select vs calib.

    Prefer roughly equal *counts* (alternating after length+noise sort) so both
    halves stay non-empty when |val|>=2; duration still influences order.
    """
    if not val_events:
        return set(), set()
    ids = sorted(val_events)
    if len(ids) == 1:
        return {ids[0]}, set()
    lengths = np.asarray([segments[i][1] - segments[i][0] for i in ids], dtype=float)
    rng = np.random.RandomState(int(nested_seed) % (2**32 - 1))
    noise = rng.random(len(ids))
    order = np.lexsort((noise, -lengths))
    select, calib = [], []
    for k, j in enumerate(order):
        eid = ids[int(j)]
        # Alternate, starting with select, then repair if counts drift by >1
        if len(select) <= len(calib):
            select.append(eid)
        else:
            calib.append(eid)
    if not calib:
        calib.append(select.pop())
    return set(select), set(calib)


def _paint_buffer_roles(
    labels: np.ndarray,
    segments: List[Tuple[int, int]],
    train_e: Set[int],
    sel_e: Set[int],
    cal_e: Set[int],
    test_e: Set[int],
    buffer: int,
) -> np.ndarray:
    """
    Point roles: 0 unused, 1 train, 2 val_select, 3 val_calib, 4 test.
    Buffer paint matches data_loader._event_level_indices priority.
    """
    role_of = {}
    for i in train_e:
        role_of[i] = 1
    for i in sel_e:
        role_of[i] = 2
    for i in cal_e:
        role_of[i] = 3
    for i in test_e:
        role_of[i] = 4

    point_role = np.zeros(len(labels), dtype=np.int8)
    for i, (s, e) in enumerate(segments):
        role = role_of[i]
        lo = max(0, s - buffer)
        hi = min(len(labels), e + buffer)
        for t in range(lo, hi):
            if point_role[t] == 0 or (s <= t < e):
                point_role[t] = role
        point_role[s:e] = role
    return point_role


def _windows_from_roles(point_role: np.ndarray, window_size: int, stride: int) -> Dict[str, np.ndarray]:
    """
    Assign windows to roles. Drop windows that mix distinct hard roles
    among {val_select, val_calib, test}. Train (1) may mix with unused (0).
    """
    n_windows = (len(point_role) - window_size) // stride + 1
    buckets = {"train": [], "val_select": [], "val_calib": [], "test_buffer": []}
    hard = {2, 3, 4}
    for wi in range(n_windows):
        a = wi * stride
        roles = set(np.unique(point_role[a:a + window_size]).tolist())
        present_hard = roles & hard
        if len(present_hard) > 1:
            continue
        if 2 in present_hard:
            buckets["val_select"].append(wi)
        elif 3 in present_hard:
            buckets["val_calib"].append(wi)
        elif 4 in present_hard:
            buckets["test_buffer"].append(wi)
        else:
            buckets["train"].append(wi)
    return {k: np.asarray(v, dtype=np.int32) for k, v in buckets.items()}


def _midpoint_territories(
    labels: np.ndarray,
    segments: List[Tuple[int, int]],
) -> np.ndarray:
    """
    For each time index, the owning event is the nearest anomaly segment
    (distance to the closed interval). Boundaries halfway between events.
    Returns event-id array of shape (T,), or -1 if no segments.
    """
    T = len(labels)
    owner = np.full(T, -1, dtype=np.int32)
    if not segments:
        return owner
    # Precompute boundaries: event i owns [left_bound, right_bound)
    bounds = []
    for i, (s, e) in enumerate(segments):
        if i == 0:
            left = 0
        else:
            prev_e = segments[i - 1][1]
            left = (prev_e + s) // 2
        if i == len(segments) - 1:
            right = T
        else:
            next_s = segments[i + 1][0]
            right = (e + next_s) // 2
        bounds.append((left, right))
        owner[left:right] = i
    return owner


def _unrestricted_test_windows(
    owner: np.ndarray,
    test_e: Set[int],
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Windows whose every point belongs to a test-assigned event territory."""
    if not test_e:
        return np.asarray([], dtype=np.int32)
    test_mask = np.isin(owner, list(test_e))
    n_windows = (len(owner) - window_size) // stride + 1
    out = []
    for wi in range(n_windows):
        a = wi * stride
        if test_mask[a:a + window_size].all():
            out.append(wi)
    return np.asarray(out, dtype=np.int32)


def _window_labels(labels: np.ndarray, indices: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    y = np.zeros(len(indices), dtype=np.int8)
    for k, wi in enumerate(indices):
        a = int(wi) * stride
        y[k] = 1 if labels[a:a + window_size].max() > 0 else 0
    return y


def _event_stats(segments: List[Tuple[int, int]], ids: Set[int]) -> Dict:
    segs = [segments[i] for i in sorted(ids)]
    return {
        "event_ids": sorted(ids),
        "n_events": len(segs),
        "anomalous_points": int(sum(e - s for s, e in segs)),
        "spans": [[int(s), int(e)] for s, e in segs],
    }


def build_partition(
    labels: np.ndarray,
    event_seed: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    buffer: int = CONTEXT_BUFFER,
) -> Dict:
    labels = (np.asarray(labels) > 0).astype(np.int8)
    segments = _anomaly_segments(labels)
    train_e, val_e, test_e = _assign_segments_by_duration(
        segments, seed=event_seed, seeded_variation=True
    )
    nested_seed = (int(event_seed) * 1_000_003 + 17) % (2**32 - 1)
    sel_e, cal_e = _split_val_events(val_e, segments, nested_seed)

    point_role = _paint_buffer_roles(labels, segments, train_e, sel_e, cal_e, test_e, buffer)
    wins = _windows_from_roles(point_role, window_size, stride)

    owner = _midpoint_territories(labels, segments)
    test_unrestricted = _unrestricted_test_windows(owner, test_e, window_size, stride)

    def prev(name, idx):
        y = _window_labels(labels, idx, window_size, stride)
        pi = float(y.mean()) if len(y) else float("nan")
        return {
            "n_windows": int(len(idx)),
            "n_pos": int(y.sum()),
            "prevalence": pi,
            "trivial_f1": (2 * pi / (1 + pi)) if pi > 0 else 0.0,
        }

    part = {
        "meta": {
            "dataset": "MSL",
            "event_seed": int(event_seed),
            "assignment": "seeded_variation=True",
            "nested_seed": nested_seed,
            "window_size": window_size,
            "stride": stride,
            "context_buffer": buffer,
            "n_events_total": len(segments),
            "protocol_version": "phaseB_v1",
        },
        "events": {
            "train": _event_stats(segments, train_e),
            "val_select": _event_stats(segments, sel_e),
            "val_calib": _event_stats(segments, cal_e),
            "test": _event_stats(segments, test_e),
        },
        "window_indices": {
            "train": wins["train"],
            "val_select": wins["val_select"],
            "val_calib": wins["val_calib"],
            "test_buffer": wins["test_buffer"],
            "test_unrestricted": test_unrestricted,
        },
        "summaries": {
            "train": prev("train", wins["train"]),
            "val_select": prev("val_select", wins["val_select"]),
            "val_calib": prev("val_calib", wins["val_calib"]),
            "test_buffer": prev("test_buffer", wins["test_buffer"]),
            "test_unrestricted": prev("test_unrestricted", test_unrestricted),
        },
    }
    return part


def save_partition(part: Dict, out_dir: str = SPLITS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    seed = part["meta"]["event_seed"]
    path = os.path.join(out_dir, f"msl_partition_es{seed}.npz")
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
    return path


def load_partition(event_seed: int, out_dir: str = SPLITS_DIR) -> Dict:
    path = os.path.join(out_dir, f"msl_partition_es{event_seed}.npz")
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


def build_scaled_dataset(
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[TimeSeriesDataset, np.ndarray, MinMaxScaler]:
    """Official-train scaler + windowed official-test series (same as get_dataloaders event)."""
    train_data, _ = load_dataset("datasets/MSL/MSL_train.npy")
    test_data, test_labels = load_dataset(
        "datasets/MSL/MSL_test.npy", "datasets/MSL/MSL_test_label.npy"
    )
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    labels = (np.asarray(test_labels) > 0).astype(np.float32)
    ds = TimeSeriesDataset(test_data, labels, window_size, stride)
    return ds, labels.astype(np.int8), scaler


def freeze_all(seeds: Sequence[int] = None, out_dir: str = SPLITS_DIR) -> Dict:
    seeds = list(seeds or EVENT_SEEDS_20)
    _, labels, _ = build_scaled_dataset()
    manifest = {"protocol_version": "phaseB_v1", "partitions": []}
    print(f"Freezing {len(seeds)} MSL partitions -> {out_dir}", flush=True)
    for es in seeds:
        part = build_partition(labels, es)
        path = save_partition(part, out_dir)
        s = part["summaries"]
        ev = part["events"]
        row = {
            "event_seed": es,
            "path": path,
            "events": {
                k: {"n_events": ev[k]["n_events"], "anomalous_points": ev[k]["anomalous_points"]}
                for k in ("train", "val_select", "val_calib", "test")
            },
            "summaries": s,
        }
        manifest["partitions"].append(row)
        print(
            f"  es={es:<5d} events tr/sel/cal/te="
            f"{ev['train']['n_events']}/{ev['val_select']['n_events']}/"
            f"{ev['val_calib']['n_events']}/{ev['test']['n_events']} | "
            f"windows train={s['train']['n_windows']} "
            f"sel={s['val_select']['n_windows']} cal={s['val_calib']['n_windows']} "
            f"buf={s['test_buffer']['n_windows']} unr={s['test_unrestricted']['n_windows']} | "
            f"pi_buf={s['test_buffer']['prevalence']:.3f} "
            f"pi_unr={s['test_unrestricted']['prevalence']:.3f}",
            flush=True,
        )
        # Guard empty calib when possible
        if s["val_calib"]["n_windows"] == 0:
            print(f"    WARNING es={es}: empty val_calib windows", flush=True)
        if s["test_buffer"]["n_windows"] == 0 or s["train"]["n_windows"] == 0:
            raise RuntimeError(f"Empty train/test_buffer for event_seed={es}")

    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default=SPLITS_DIR)
    args = ap.parse_args()
    freeze_all(args.seeds, args.out_dir)


if __name__ == "__main__":
    main()
