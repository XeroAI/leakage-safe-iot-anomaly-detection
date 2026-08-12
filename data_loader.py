import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, window_size=100, stride=1):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.num_windows = (len(self.data) - self.window_size) // self.stride + 1

    def __len__(self):
        return self.num_windows

    def window_label(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        window_label = self.labels[start_idx:end_idx]
        return 1 if np.max(window_label) > 0 else 0

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        window_data = self.data[start_idx:end_idx]
        final_label = float(self.window_label(idx))

        tensor_data = torch.tensor(window_data, dtype=torch.float32).transpose(0, 1)
        tensor_label = torch.tensor(final_label, dtype=torch.float32)

        return tensor_data, tensor_label


def load_dataset(data_path, label_path=None):
    data = np.load(data_path)
    labels = None if label_path is None else np.load(label_path)

    if labels is not None and len(labels.shape) > 1:
        labels = labels.squeeze(-1)

    return data, labels


def _block_indices(n_windows, window_size, stride, ratios=(0.40, 0.20, 0.40)):
    """
    Contiguous temporal blocks on ordered window indices with boundary gaps.

    ratios apply to the usable length after reserving gap windows between blocks.
    """
    gap = max(1, window_size // stride)
    r0, r1, r2 = ratios
    usable = n_windows - 2 * gap
    if usable < 3:
        n_a = max(1, n_windows // 3)
        n_b = max(1, n_windows // 3)
        a = np.arange(0, n_a)
        b = np.arange(n_a, n_a + n_b)
        c = np.arange(n_a + n_b, n_windows)
        return a, b, c

    n_a = max(1, int(round(usable * r0)))
    n_b = max(1, int(round(usable * r1)))
    n_c = max(1, usable - n_a - n_b)

    a_end = n_a
    b_start = a_end + gap
    b_end = b_start + n_b
    c_start = b_end + gap
    c_end = min(n_windows, c_start + n_c)

    return (
        np.arange(0, a_end),
        np.arange(b_start, b_end),
        np.arange(c_start, c_end),
    )


def _chronological_indices(n_windows, window_size, stride, train_ratio=0.70, val_ratio=0.15):
    """Legacy chronological blocks on a single concatenated timeline."""
    return _block_indices(
        n_windows, window_size, stride, ratios=(train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)
    )


def _interleaved_block_indices(n_windows, window_size, stride, n_blocks=12):
    """
    Leakage-safe split that reduces early/late anomaly mismatch.

    Divide ordered windows into ``n_blocks`` contiguous blocks, drop a gap of
    ``window_size // stride`` windows at each block edge (so adjacent splits do
    not share overlapping windows), then assign blocks round-robin to
    train / val / test.
    """
    if n_blocks < 3:
        raise ValueError("n_blocks must be >= 3")
    gap = max(1, window_size // stride)
    block_size = n_windows // n_blocks
    train_idx, val_idx, test_idx = [], [], []

    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else n_windows
        # Shrink block interior to avoid cross-block window overlap.
        inner_start = start + gap
        inner_end = end - gap
        if inner_end <= inner_start:
            idxs = np.arange(start, end)
        else:
            idxs = np.arange(inner_start, inner_end)

        role = b % 3
        if role == 0:
            train_idx.append(idxs)
        elif role == 1:
            val_idx.append(idxs)
        else:
            test_idx.append(idxs)

    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=int),
        np.concatenate(val_idx) if val_idx else np.array([], dtype=int),
        np.concatenate(test_idx) if test_idx else np.array([], dtype=int),
    )


def _stratified_indices(window_labels, train_ratio=0.70):
    """Legacy random stratified split (kept for ablation / comparison only)."""
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(window_labels)), window_labels))

    temp_labels = window_labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx_rel, test_idx_rel = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    return train_idx, val_idx, test_idx


def _anomaly_segments(labels):
    """Contiguous anomaly intervals as (start, end) half-open index pairs."""
    lab = np.asarray(labels).astype(int).ravel()
    padded = np.concatenate([[0], lab > 0, [0]]).astype(int)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _assign_segments_by_duration(
    segments,
    train_ratio=0.50,
    val_ratio=0.25,
    seed=42,
    seeded_variation=False,
):
    """
    Assign anomaly events to train/val/test, balancing total anomaly duration.

    Default (``seeded_variation=False``): longest-first greedy load balancing.
    The RNG seed is consumed for determinism but does not change the partition;
    this preserves the published MSL event split at ``seed=42``.

    With ``seeded_variation=True``: equal-length ordering and equal load/target
    ties are broken with a seeded RNG so distinct ``seed`` values yield distinct
    admissible partitions (event-split variance studies).
    """
    if not segments:
        return set(), set(), set()

    lengths = np.asarray([e - s for s, e in segments], dtype=float)
    rng = np.random.RandomState(seed)
    load = {"train": 0.0, "val": 0.0, "test": 0.0}
    targets = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(1.0 - train_ratio - val_ratio),
    }
    buckets = {"train": [], "val": [], "test": []}

    if not seeded_variation:
        _ = rng.randint(0, 1_000_000)
        order = np.argsort(lengths)[::-1]
        for i in order:
            best = min(load.keys(), key=lambda k: load[k] / max(targets[k], 1e-9))
            buckets[best].append(int(i))
            load[best] += float(lengths[i])
        return set(buckets["train"]), set(buckets["val"]), set(buckets["test"])

    noise = rng.random(len(lengths))
    order = np.lexsort((noise, -lengths))
    split_names = ("train", "val", "test")
    for i in order:
        scores = {k: load[k] / max(targets[k], 1e-9) for k in split_names}
        best_score = min(scores.values())
        tied = [k for k, v in scores.items() if abs(v - best_score) < 1e-12]
        best = tied[int(rng.randint(0, len(tied)))] if len(tied) > 1 else tied[0]
        buckets[best].append(int(i))
        load[best] += float(lengths[i])

    return set(buckets["train"]), set(buckets["val"]), set(buckets["test"])


def _event_level_indices(
    labels,
    window_size,
    stride,
    context_buffer=200,
    seed=42,
    seeded_variation=False,
):
    """
    Leakage-safe supervised protocol: hold out whole anomaly *events*.

    Contiguous anomaly segments on the official test labels are assigned to
    train/val/test (duration-balanced). A context buffer of normal timesteps
    around each segment follows the same assignment so val/test contain both
    positives and negatives. Windows that mix val and test roles are dropped.
    No anomaly event appears in more than one split.
    """
    segments = _anomaly_segments(labels)
    train_segs, val_segs, test_segs = _assign_segments_by_duration(
        segments, seed=seed, seeded_variation=seeded_variation
    )

    point_role = np.zeros(len(labels), dtype=np.int8)  # 0 unused, 1 train, 2 val, 3 test
    for i, (s, e) in enumerate(segments):
        role = 1 if i in train_segs else (2 if i in val_segs else 3)
        lo = max(0, s - context_buffer)
        hi = min(len(labels), e + context_buffer)
        for t in range(lo, hi):
            if point_role[t] == 0 or (s <= t < e):
                point_role[t] = role
        point_role[s:e] = role

    n_windows = (len(labels) - window_size) // stride + 1
    train_idx, val_idx, test_idx = [], [], []
    for wi in range(n_windows):
        a = wi * stride
        b = a + window_size
        roles = set(np.unique(point_role[a:b]).tolist())
        hard = roles - {0, 1}
        if 2 in hard and 3 in hard:
            continue
        if 2 in hard:
            val_idx.append(wi)
        elif 3 in hard:
            test_idx.append(wi)
        else:
            train_idx.append(wi)

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def load_smd_machine(machine_id, base_dir="datasets/SMD"):
    """Load one SMD machine from official train/test/test_label text files."""
    train_file = os.path.join(base_dir, "train", f"{machine_id}.txt")
    test_file = os.path.join(base_dir, "test", f"{machine_id}.txt")
    test_label_file = os.path.join(base_dir, "test_label", f"{machine_id}.txt")

    for path in (train_file, test_file, test_label_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing SMD file: {path}")

    train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(test_label_file, delimiter=",", dtype=np.float32)
    if len(test_labels.shape) > 1:
        test_labels = test_labels.squeeze(-1)
    return train_data, test_data, test_labels


def get_dataloaders(
    dataset_name="MSL",
    batch_size=32,
    window_size=100,
    stride=50,
    split_mode="event",
    machine_id=None,
    smd_base_dir="datasets/SMD",
    event_seed=42,
    seeded_variation=False,
):
    """
    Build train/val/test loaders.

    split_mode:
      - \"event\" (default): leakage-safe event-level holdout on official test.
            * MinMax fitted on official train only.
            * Windows built inside the official test sequence.
            * Whole anomaly segments assigned to train/val/test (duration-balanced)
              with local normal context; no event is shared across splits.
            * Official-train normals are NOT mixed into the supervised train set
              (they only fit the scaler), to avoid extreme class dilution.
      - \"interleaved\": contiguous test blocks assigned round-robin (with gaps).
      - \"official\": early/mid/late blocks on official test (harsh temporal shift).
      - \"chronological\": windows on train||test concatenation, then temporal blocks.
      - \"stratified\": legacy random stratified window split (leaky; not recommended).

    machine_id:
      When dataset_name is \"SMD\", load per-machine official train/test files
      instead of the aggregated SMD_*.npy arrays.

    seeded_variation:
      If True with split_mode=\"event\", distinct event_seed values produce distinct
      partitions (for multi-split variance). Default False preserves published splits.
    """
    base_path = f"datasets/{dataset_name}"

    if dataset_name == "MSL":
        train_path = os.path.join(base_path, "MSL_train.npy")
        test_path = os.path.join(base_path, "MSL_test.npy")
        test_label_path = os.path.join(base_path, "MSL_test_label.npy")
        train_data, _ = load_dataset(train_path)
        test_data, test_labels = load_dataset(test_path, test_label_path)
    elif dataset_name == "SMD":
        if machine_id:
            train_data, test_data, test_labels = load_smd_machine(machine_id, smd_base_dir)
        else:
            train_path = os.path.join(base_path, "SMD_train.npy")
            test_path = os.path.join(base_path, "SMD_test.npy")
            test_label_path = os.path.join(base_path, "SMD_test_label.npy")
            train_data, _ = load_dataset(train_path)
            test_data, test_labels = load_dataset(test_path, test_label_path)
    else:
        raise ValueError("Dataset name must be MSL or SMD")

    # Fit normalization on official train partition ONLY
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    if split_mode in ("event", "interleaved", "official"):
        # Window inside each official partition (no cross-partition overlap)
        train_labels = np.zeros(len(train_data), dtype=np.float32)
        train_ds = TimeSeriesDataset(train_data, train_labels, window_size, stride)
        test_ds = TimeSeriesDataset(test_data, test_labels, window_size, stride)

        if split_mode == "event":
            labeled_train_idx, val_idx, test_idx = _event_level_indices(
                test_labels,
                window_size,
                stride,
                context_buffer=max(window_size, 200),
                seed=event_seed,
                seeded_variation=seeded_variation,
            )
            # Event protocol: train only on official-test windows whose anomaly
            # events (and local context) are assigned to train. Official-train
            # normals are used solely to fit the scaler — concatenating all of
            # them dilutes positives and hurts transfer to held-out events.
            train_dataset = Subset(test_ds, labeled_train_idx)
            val_dataset = Subset(test_ds, val_idx)
            test_dataset = Subset(test_ds, test_idx)
        else:
            if split_mode == "interleaved":
                labeled_train_idx, val_idx, test_idx = _interleaved_block_indices(
                    len(test_ds), window_size, stride, n_blocks=12
                )
            else:
                # Harsh early/mid/late protocol (kept for sensitivity analysis)
                labeled_train_idx, val_idx, test_idx = _block_indices(
                    len(test_ds), window_size, stride, ratios=(0.40, 0.20, 0.40)
                )

            # Supervised training: official-train normals + labeled train windows
            labeled_train = Subset(test_ds, labeled_train_idx)
            train_dataset = ConcatDataset([train_ds, labeled_train])
            val_dataset = Subset(test_ds, val_idx)
            test_dataset = Subset(test_ds, test_idx)

    elif split_mode == "chronological":
        all_data = np.vstack([train_data, test_data])
        all_labels = np.concatenate([np.zeros(len(train_data)), test_labels])
        full_dataset = TimeSeriesDataset(all_data, all_labels, window_size, stride)
        train_idx, val_idx, test_idx = _chronological_indices(
            len(full_dataset), window_size, stride
        )
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)

    elif split_mode == "stratified":
        all_data = np.vstack([train_data, test_data])
        all_labels = np.concatenate([np.zeros(len(train_data)), test_labels])
        full_dataset = TimeSeriesDataset(all_data, all_labels, window_size, stride)
        all_window_labels = np.array(
            [full_dataset.window_label(i) for i in range(len(full_dataset))]
        )
        train_idx, val_idx, test_idx = _stratified_indices(all_window_labels)
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)
    else:
        raise ValueError(
            "split_mode must be 'event', 'interleaved', 'official', "
            "'chronological', or 'stratified'"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing Data Loader (official-partition-aware, leakage-safe)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name="MSL", batch_size=32, window_size=100, stride=10,
        split_mode="official",
    )

    def split_stats(loader, name):
        labels = []
        for _, target in loader:
            labels.extend(target.numpy())
        labels = np.array(labels)
        anomaly_ratio = labels.mean() * 100 if len(labels) else 0.0
        print(
            f"{name}: {len(loader.dataset)} windows, "
            f"anomaly ratio: {anomaly_ratio:.1f}%"
        )

    split_stats(train_loader, "Train")
    split_stats(val_loader, "Val")
    split_stats(test_loader, "Test")

    batch_data, batch_labels = next(iter(train_loader))
    print(f"Data batch shape: {batch_data.shape}")
