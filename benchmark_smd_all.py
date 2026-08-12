import argparse
import glob
import os

import numpy as np

from preprocess_smd import preprocess_smd
from train import EPOCHS, train_model


def list_smd_machines(base_dir="datasets/SMD"):
    train_files = sorted(glob.glob(os.path.join(base_dir, "train", "*.txt")))
    return [os.path.basename(path).replace(".txt", "") for path in train_files]


def benchmark_all_machines(epochs=EPOCHS, max_machines=None):
    machines = list_smd_machines()
    if max_machines is not None:
        machines = machines[:max_machines]

    print(f"Benchmarking {len(machines)} SMD machines ({epochs} epochs each)...")
    print("=" * 90)

    results = []
    for idx, machine_id in enumerate(machines, start=1):
        print(f"\n[{idx}/{len(machines)}] {machine_id}")
        preprocess_smd(machine_id=machine_id)

        metrics = train_model(
            epochs=epochs,
            dataset_name="SMD",
            model_save_path=f"lst_tfdn_SMD_{machine_id}_best.pth",
            verbose=False,
        )

        row = {
            "machine": machine_id,
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "best_val_f1": metrics["best_val_f1"],
        }
        results.append(row)
        print(
            f"  Test F1: {row['test_f1']:.4f} | "
            f"Precision: {row['test_precision']:.4f} | "
            f"Recall: {row['test_recall']:.4f}"
        )

    f1_scores = np.array([r["test_f1"] for r in results])
    prec_scores = np.array([r["test_precision"] for r in results])
    rec_scores = np.array([r["test_recall"] for r in results])

    print("\n" + "=" * 90)
    print("SMD FULL BENCHMARK SUMMARY (28 machines, mean +/- std)")
    print("=" * 90)
    print(f"Test F1:        {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
    print(f"Test Precision: {prec_scores.mean():.4f} +/- {prec_scores.std():.4f}")
    print(f"Test Recall:    {rec_scores.mean():.4f} +/- {rec_scores.std():.4f}")
    print("\nPer-machine results:")
    for row in results:
        print(
            f"  {row['machine']}: F1={row['test_f1']:.4f}, "
            f"P={row['test_precision']:.4f}, R={row['test_recall']:.4f}"
        )

    save_results_to_file(
        results,
        f1_scores.mean(),
        f1_scores.std(),
        prec_scores.mean(),
        prec_scores.std(),
        rec_scores.mean(),
        rec_scores.std(),
    )

    return results


def save_results_to_file(
    results, mean_f1, std_f1, mean_p, std_p, mean_r, std_r, path="benchmark_smd_all_results.txt"
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("SMD FULL BENCHMARK SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Test F1:        {mean_f1:.4f} +/- {std_f1:.4f}\n")
        f.write(f"Test Precision: {mean_p:.4f} +/- {std_p:.4f}\n")
        f.write(f"Test Recall:    {mean_r:.4f} +/- {std_r:.4f}\n\n")
        f.write("Per-machine results:\n")
        for row in results:
            f.write(
                f"  {row['machine']}: F1={row['test_f1']:.4f}, "
                f"P={row['test_precision']:.4f}, R={row['test_recall']:.4f}\n"
            )
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate LST-TFDN on all SMD machines."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--max-machines",
        type=int,
        default=None,
        help="Limit number of machines (useful for quick tests).",
    )
    args = parser.parse_args()

    benchmark_all_machines(epochs=args.epochs, max_machines=args.max_machines)
