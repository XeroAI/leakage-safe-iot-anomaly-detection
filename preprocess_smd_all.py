import glob
import os

import numpy as np


def preprocess_all_smd(base_dir="datasets/SMD"):
    train_dir = os.path.join(base_dir, "train")
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))

    if len(train_files) == 0:
        print(f"ERROR: No .txt files found in {train_dir}")
        return

    print(f"Found {len(train_files)} machines. Processing...")

    all_train = []
    all_test = []
    all_labels = []

    for train_file in train_files:
        machine_id = os.path.basename(train_file).replace(".txt", "")
        test_file = os.path.join(base_dir, "test", f"{machine_id}.txt")
        test_label_file = os.path.join(base_dir, "test_label", f"{machine_id}.txt")

        if not os.path.exists(test_file) or not os.path.exists(test_label_file):
            print(f"WARNING: Skipping {machine_id} (missing test or label file)")
            continue

        train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
        test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
        test_labels = np.loadtxt(test_label_file, delimiter=",", dtype=np.float32)

        if len(test_labels.shape) > 1:
            test_labels = test_labels.squeeze(-1)

        all_train.append(train_data)
        all_test.append(test_data)
        all_labels.append(test_labels)
        print(f"  Loaded {machine_id}: train {train_data.shape}, test {test_data.shape}")

    train_data = np.vstack(all_train)
    test_data = np.vstack(all_test)
    test_labels = np.concatenate(all_labels)

    print(f"Aggregated train shape: {train_data.shape}")
    print(f"Aggregated test shape: {test_data.shape}")
    print(f"Aggregated labels shape: {test_labels.shape}")

    np.save(os.path.join(base_dir, "SMD_train.npy"), train_data)
    np.save(os.path.join(base_dir, "SMD_test.npy"), test_data)
    np.save(os.path.join(base_dir, "SMD_test_label.npy"), test_labels)

    print(
        "Successfully saved aggregated SMD_train.npy, "
        "SMD_test.npy, and SMD_test_label.npy!"
    )


if __name__ == "__main__":
    preprocess_all_smd()
