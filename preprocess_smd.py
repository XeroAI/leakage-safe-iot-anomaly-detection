import os

import numpy as np


def preprocess_smd(machine_id="machine-1-1", base_dir="datasets/SMD"):
    train_file = os.path.join(base_dir, "train", f"{machine_id}.txt")
    test_file = os.path.join(base_dir, "test", f"{machine_id}.txt")
    test_label_file = os.path.join(base_dir, "test_label", f"{machine_id}.txt")

    print(f"Loading SMD data for {machine_id}...")

    for path in (train_file, test_file, test_label_file):
        if not os.path.exists(path):
            print(f"ERROR: Could not find {path}")
            print(
                "Please ensure datasets/SMD contains train/, test/, and "
                "test_label/ subfolders with .txt files."
            )
            return

    train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(test_label_file, delimiter=",", dtype=np.float32)

    if len(test_labels.shape) > 1:
        test_labels = test_labels.squeeze(-1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    np.save(os.path.join(base_dir, "SMD_train.npy"), train_data)
    np.save(os.path.join(base_dir, "SMD_test.npy"), test_data)
    np.save(os.path.join(base_dir, "SMD_test_label.npy"), test_labels)

    print("Successfully saved SMD_train.npy, SMD_test.npy, and SMD_test_label.npy!")


if __name__ == "__main__":
    preprocess_smd()
