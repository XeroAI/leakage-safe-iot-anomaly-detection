"""
Repeated-seed evaluation for LST-TFDN under the official-partition-aware protocol.
Reports mean +/- std and approximate 95% CI across seeds.
"""

import argparse
import json

import numpy as np

from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MSL")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split_mode", default="event")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"\n===== Seed {seed} =====", flush=True)
        metrics = train_model(
            epochs=args.epochs,
            dataset_name=args.dataset,
            model_save_path=f"lst_tfdn_{args.dataset}_seed{seed}.pth",
            verbose=True,
            seed=seed,
            split_mode=args.split_mode,
        )
        # train_model currently may return None — capture from prints if needed.
        rows.append({"seed": seed, "metrics": metrics})

    # If train_model returns a dict, aggregate; otherwise write seed list for the user.
    numeric = [r["metrics"] for r in rows if isinstance(r["metrics"], dict) and "test_f1" in r["metrics"]]
    summary = {"dataset": args.dataset, "split_mode": args.split_mode, "seeds": args.seeds}
    if numeric:
        f1s = np.array([m["test_f1"] for m in numeric], dtype=float)
        summary["f1_mean"] = float(f1s.mean())
        summary["f1_std"] = float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0
        summary["f1_ci95"] = float(1.96 * summary["f1_std"] / max(len(f1s), 1) ** 0.5)
        print(
            f"\nF1 mean={summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f} "
            f"(approx. 95% CI half-width {summary['f1_ci95']:.4f})"
        )

    out_path = f"multi_seed_{args.dataset}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
