"""
SMD event-level benchmark: 28 machines × 3 models × 3 seeds.

Uses the same leakage-safe event-level holdout protocol as MSL:
  - scaler fit on official per-machine train (normal-only)
  - supervised windows from official test labels only
  - whole anomaly events held out across train/val/test
"""

import argparse
import json
import os

import numpy as np

from baselines_event import (
    BASELINES,
    BATCH_SIZE,
    DEFAULT_SEEDS,
    EPOCHS,
    SPLIT_MODE,
    STRIDE,
    WINDOW_SIZE,
    count_params,
    set_seed,
    train_and_evaluate,
)
from benchmark_smd_all import list_smd_machines
from data_loader import get_dataloaders


RESULTS_PATH = "smd_event_benchmark_results.json"


def run_key(machine_id, model_name, seed):
    return f"{machine_id}|{model_name}|{seed}"


def load_existing_runs(path=RESULTS_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("runs", [])


def save_runs(runs, summary=None, path=RESULTS_PATH):
    payload = {"runs": runs}
    if summary is not None:
        payload["summary"] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_runs(runs, baselines=None):
    baselines = baselines or BASELINES
    summary = {
        "split_mode": SPLIT_MODE,
        "n_machines": len({r["machine"] for r in runs}),
        "seeds": sorted({r["seed"] for r in runs}),
        "models": [],
    }
    for name, _ in baselines:
        rows = [r for r in runs if r["name"] == name]
        if not rows:
            continue
        f1 = np.array([r["test_f1"] for r in rows], dtype=float)
        prec = np.array([r["test_precision"] for r in rows], dtype=float)
        rec = np.array([r["test_recall"] for r in rows], dtype=float)
        summary["models"].append(
            {
                "name": name,
                "num_params": rows[0]["num_params"],
                "inference_ms": float(np.mean([r["inference_ms"] for r in rows])),
                "f1_mean": float(f1.mean()),
                "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "precision_mean": float(prec.mean()),
                "recall_mean": float(rec.mean()),
                "n_runs": len(rows),
            }
        )
    return summary


def print_summary(summary):
    print("\n" + "=" * 100)
    print("SMD EVENT-LEVEL BENCHMARK SUMMARY (28 machines, 3 seeds per model)")
    print("=" * 100)
    print(
        f"{'Model':<28} {'Test F1':>14} {'Precision':>10} {'Recall':>10} "
        f"{'Params':>12} {'Inf (ms)':>10} {'Runs':>6}"
    )
    print("-" * 100)
    for m in summary["models"]:
        f1_str = f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
        print(
            f"{m['name']:<28} {f1_str:>14} {m['precision_mean']:>10.4f} {m['recall_mean']:>10.4f} "
            f"{m['num_params']:>12,} {m['inference_ms']:>10.1f} {m['n_runs']:>6}"
        )
    print("=" * 100)


def save_txt(summary, path="smd_event_benchmark_results.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("SMD Event-Level Benchmark (per-machine official train/test, event holdout)\n")
        f.write("=" * 100 + "\n")
        for m in summary["models"]:
            f.write(
                f"{m['name']}: F1={m['f1_mean']:.4f}±{m['f1_std']:.4f}, "
                f"P={m['precision_mean']:.4f}, R={m['recall_mean']:.4f}, "
                f"Params={m['num_params']:,}, Inf={m['inference_ms']:.1f}ms, runs={m['n_runs']}\n"
            )
        f.write("\nLaTeX rows:\n")
        for m in summary["models"]:
            f.write(
                f"{m['name']} & ${m['f1_mean']:.4f} \\pm {m['f1_std']:.4f}$ & "
                f"{m['precision_mean']:.4f} & {m['recall_mean']:.4f} & "
                f"{m['num_params']:,} & {m['inference_ms']:.1f} \\\\\n"
            )
    print(f"Saved {path}")


def run_benchmark(epochs=EPOCHS, seeds=None, max_machines=None, baselines=None, verbose=True):
    import torch

    seeds = seeds or DEFAULT_SEEDS
    baselines = baselines or BASELINES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machines = list_smd_machines()
    if max_machines is not None:
        machines = machines[:max_machines]

    runs = load_existing_runs()
    done = {run_key(r["machine"], r["name"], r["seed"]) for r in runs}

    print("=" * 100)
    print(
        f"SMD event-level benchmark | machines={len(machines)} | models={len(baselines)} | "
        f"seeds={seeds} | epochs={epochs} | device={device}"
    )
    print("=" * 100)

    for mi, machine_id in enumerate(machines, start=1):
        print(f"\n[{mi}/{len(machines)}] {machine_id}")
        train_loader, val_loader, test_loader = get_dataloaders(
            "SMD",
            BATCH_SIZE,
            WINDOW_SIZE,
            stride=STRIDE,
            split_mode=SPLIT_MODE,
            machine_id=machine_id,
        )
        num_sensors = next(iter(train_loader))[0].shape[1]

        if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print(f"  Skipping {machine_id}: empty train or test split")
            continue

        for model_name, build_model in baselines:
            num_params = count_params(build_model(num_sensors, WINDOW_SIZE))
            for seed in seeds:
                key = run_key(machine_id, model_name, seed)
                if key in done:
                    print(f"  Skip {model_name} seed {seed} (already done)")
                    continue

                set_seed(seed)
                model = build_model(num_sensors, WINDOW_SIZE).to(device)
                if verbose:
                    print(f"  {model_name} | seed {seed} | params {num_params:,}")
                metrics = train_and_evaluate(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    epochs,
                    verbose=verbose,
                )
                row = {
                    "machine": machine_id,
                    "name": model_name,
                    "seed": seed,
                    "num_params": num_params,
                    **metrics,
                }
                runs.append(row)
                done.add(key)
                save_runs(runs)
                print(
                    f"    -> F1={row['test_f1']:.4f} P={row['test_precision']:.4f} "
                    f"R={row['test_recall']:.4f} Inf={row['inference_ms']:.1f}ms"
                )

    summary = summarize_runs(runs, baselines)
    save_runs(runs, summary=summary)
    print_summary(summary)
    save_txt(summary)
    return runs, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMD event-level benchmark (28 machines).")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-machines", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 machine, 3 epochs, seed 42, LST-TFDN only.",
    )
    args = parser.parse_args()

    if args.quick:
        import baselines_event as be

        run_benchmark(
            epochs=3,
            seeds=[42],
            max_machines=1,
            baselines=[be.BASELINES[0]],
            verbose=True,
        )
    else:
        run_benchmark(
            epochs=args.epochs,
            seeds=args.seeds,
            max_machines=args.max_machines,
            verbose=True,
        )
