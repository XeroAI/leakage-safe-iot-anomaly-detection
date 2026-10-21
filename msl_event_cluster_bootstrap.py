"""
Event-cluster / hierarchical bootstrap on multi-partition retained scores (M2, M3).

Reads msl_multipartition/scores/*.npz and frozen splits.
Writes msl_multipartition/bootstrap_results.{json,txt}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loader import _anomaly_segments, load_dataset
from msl_event_splits import CONTEXT_BUFFER, STRIDE, WINDOW_SIZE, load_partition

ROOT = "msl_multipartition"
SCORES = os.path.join(ROOT, "scores")
OUT_JSON = os.path.join(ROOT, "bootstrap_results.json")
OUT_TXT = os.path.join(ROOT, "bootstrap_results.txt")
MODELS = ["CNN-only", "LST-TFDN", "Standard Transformer"]
METRICS = ["f1", "delta_f1", "auroc", "auprc_norm", "mcc"]


def metrics_from(y, p, thr) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > thr).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    pi = float(y.mean()) if len(y) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    trivial = 2 * pi / (1 + pi) if pi > 0 else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    if 0 < y.sum() < len(y):
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
    else:
        auroc = auprc = float("nan")
    return {
        "f1": f1,
        "delta_f1": f1 - trivial,
        "auroc": auroc,
        "auprc_norm": auprc / pi if pi > 0 else float("nan"),
        "mcc": mcc,
        "trivial_f1": trivial,
        "prevalence": pi,
    }


def bca(boot, theta, jack, alpha=0.05):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    jack = np.asarray(jack, float)
    jack = jack[np.isfinite(jack)]
    if len(boot) < 10:
        return float("nan"), float("nan")
    prop = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    prop = min(max(prop, 1 / (len(boot) + 1)), 1 - 1 / (len(boot) + 1))
    z0 = norm.ppf(prop)
    if len(jack) >= 3:
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        acc = num / den if den > 0 else 0.0
    else:
        acc = 0.0
    out = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        q = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.quantile(boot, q)))
    return out[0], out[1]


def buffer_clusters(labels, test_idx, event_ids_test, segments) -> np.ndarray:
    """Map each buffer test window to a connected test-role component id."""
    # paint test events + buffer
    role = np.zeros(len(labels), dtype=np.int8)
    test_set = set(event_ids_test)
    for i, (s, e) in enumerate(segments):
        if i not in test_set:
            continue
        lo = max(0, s - CONTEXT_BUFFER)
        hi = min(len(labels), e + CONTEXT_BUFFER)
        role[lo:hi] = 1
        role[s:e] = 1
    padded = np.concatenate([[0], role, [0]])
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    comp = np.full(len(labels), -1, dtype=int)
    for c, (a, b) in enumerate(zip(starts, ends)):
        comp[a:b] = c
    out = np.empty(len(test_idx), dtype=int)
    for k, wi in enumerate(test_idx):
        ids = np.unique(comp[int(wi) * STRIDE: int(wi) * STRIDE + WINDOW_SIZE])
        ids = ids[ids >= 0]
        out[k] = int(ids[0]) if len(ids) else -1
    return out


def load_deep_runs():
    pat = re.compile(r"deep_(.+)_es(\d+)_init(\d+)\.npz$")
    runs = {}
    for path in sorted(glob.glob(os.path.join(SCORES, "deep_*.npz"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        name = m.group(1).replace("_", " ")
        es, ini = int(m.group(2)), int(m.group(3))
        z = np.load(path)
        runs[(name, es, ini)] = {
            "probs": z["probs_buffer"].astype(float),
            "targets": z["targets_buffer"].astype(int),
            "widx": z["window_indices_buffer"].astype(int),
            "thr": float(z["thresh"][0]),
            "probs_unr": z["probs_unrestricted"].astype(float),
            "targets_unr": z["targets_unrestricted"].astype(int),
            "path": path,
        }
    return runs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    _, labels = load_dataset("datasets/MSL/MSL_test.npy", "datasets/MSL/MSL_test_label.npy")
    labels = (labels > 0).astype(int)
    segments = _anomaly_segments(labels)
    runs = load_deep_runs()
    print(f"Loaded {len(runs)} deep buffer score files", flush=True)

    # attach clusters from frozen partitions
    cluster_cache = {}
    for (name, es, ini), r in runs.items():
        if es not in cluster_cache:
            part = load_partition(es)
            te_ids = set(part["events"]["test"]["event_ids"])
            # map run widx order to clusters
            cl = buffer_clusters(labels, part["window_indices"]["test_buffer"], te_ids, segments)
            # align: scores saved in subset order = partition test_buffer order
            cluster_cache[es] = {
                "widx": part["window_indices"]["test_buffer"],
                "cluster": cl,
                "events": part["events"],
                "summaries": part["summaries"],
            }
        ref = cluster_cache[es]
        if not np.array_equal(r["widx"], ref["widx"]):
            # reorder
            pos = {int(w): i for i, w in enumerate(ref["widx"])}
            order = np.asarray([pos[int(w)] for w in r["widx"]])
            r["cluster"] = ref["cluster"][order]
        else:
            r["cluster"] = ref["cluster"]

    event_seeds = sorted({k[1] for k in runs})
    init_seeds = sorted({k[2] for k in runs})

    per_run = []
    for key in sorted(runs):
        name, es, ini = key
        r = runs[key]
        y, p, thr = r["targets"], r["probs"], r["thr"]
        groups = [np.where(r["cluster"] == c)[0] for c in np.unique(r["cluster"]) if c >= 0]
        point = metrics_from(y, p, thr)
        boot = defaultdict(list)
        for _ in range(args.reps):
            pick = [groups[j] for j in rng.randint(0, len(groups), size=len(groups))]
            idx = np.concatenate(pick)
            m = metrics_from(y[idx], p[idx], thr)
            for k in METRICS:
                boot[k].append(m[k])
        jack = defaultdict(list)
        for g in range(len(groups)):
            keep = np.concatenate([groups[j] for j in range(len(groups)) if j != g])
            m = metrics_from(y[keep], p[keep], thr)
            for k in METRICS:
                jack[k].append(m[k])
        # unrestricted point (no cluster boot here; descriptive)
        unr = metrics_from(r["targets_unr"], r["probs_unr"], thr)
        entry = {
            "model": name, "event_seed": es, "init_seed": ini,
            "n_windows": len(y), "n_clusters": len(groups),
            "buffer": point,
            "unrestricted": unr,
            "cluster_bca": {},
        }
        for k in METRICS:
            lo, hi = bca(boot[k], point[k], jack[k])
            entry["cluster_bca"][k] = {
                "point": point[k],
                "bca_lo": lo, "bca_hi": hi,
                "pct_lo": float(np.nanquantile(boot[k], 0.025)),
                "pct_hi": float(np.nanquantile(boot[k], 0.975)),
            }
        per_run.append(entry)
        print(
            f"{name:10s} es={es:<5d} init={ini:<4d} "
            f"dF1={point['delta_f1']:+.3f} [{entry['cluster_bca']['delta_f1']['bca_lo']:+.3f},"
            f"{entry['cluster_bca']['delta_f1']['bca_hi']:+.3f}] "
            f"AUROC={point['auroc']:.3f} [{entry['cluster_bca']['auroc']['bca_lo']:.3f},"
            f"{entry['cluster_bca']['auroc']['bca_hi']:.3f}]",
            flush=True,
        )

    # hierarchical across partitions
    across = []
    for name in MODELS:
        part_means = {m: [] for m in METRICS}
        part_unr = {m: [] for m in METRICS}
        for es in event_seeds:
            keys = [(name, es, i) for i in init_seeds if (name, es, i) in runs]
            if not keys:
                continue
            for m in METRICS:
                part_means[m].append(float(np.mean([
                    metrics_from(runs[k]["targets"], runs[k]["probs"], runs[k]["thr"])[m] for k in keys
                ])))
                part_unr[m].append(float(np.mean([
                    metrics_from(runs[k]["targets_unr"], runs[k]["probs_unr"], runs[k]["thr"])[m] for k in keys
                ])))
        # bootstrap partitions x inits x clusters
        boot = defaultdict(list)
        es_list = [es for es in event_seeds if any((name, es, i) in runs for i in init_seeds)]
        for _ in range(args.reps):
            pick_es = [es_list[j] for j in rng.randint(0, len(es_list), size=len(es_list))]
            vals = defaultdict(list)
            for es in pick_es:
                keys = [(name, es, i) for i in init_seeds if (name, es, i) in runs]
                seeds = [keys[j] for j in rng.randint(0, len(keys), size=len(keys))]
                groups = [np.where(runs[keys[0]]["cluster"] == c)[0]
                          for c in np.unique(runs[keys[0]]["cluster"]) if c >= 0]
                idx = np.concatenate([groups[j] for j in rng.randint(0, len(groups), size=len(groups))])
                ms = [metrics_from(runs[k]["targets"][idx], runs[k]["probs"][idx], runs[k]["thr"]) for k in seeds]
                for m in METRICS:
                    vals[m].append(float(np.mean([x[m] for x in ms])))
            for m in METRICS:
                boot[m].append(float(np.mean(vals[m])))
        row = {
            "model": name,
            "n_partitions": len(es_list),
            "per_partition_buffer": part_means,
            "per_partition_unrestricted": part_unr,
            "above_trivial_buffer": int(sum(1 for x in part_means["delta_f1"] if x > 0)),
            "summary_buffer": {
                m: {
                    "point": float(np.mean(part_means[m])),
                    "pct_lo": float(np.nanquantile(boot[m], 0.025)),
                    "pct_hi": float(np.nanquantile(boot[m], 0.975)),
                } for m in METRICS
            },
        }
        across.append(row)

    # HistGB across partitions (point only + partition bootstrap)
    hist_rows = []
    for path in sorted(glob.glob(os.path.join(SCORES, "histgb_fixed_es*.npz"))):
        es = int(re.search(r"es(\d+)", path).group(1))
        z = np.load(path)
        thr = float(z["thresh"][0])
        mb = metrics_from(z["targets_buffer"], z["probs_buffer"], thr)
        mu = metrics_from(z["targets_unrestricted"], z["probs_unrestricted"], thr)
        hist_rows.append({"event_seed": es, "buffer": mb, "unrestricted": mu})

    out = {
        "meta": {"reps": args.reps, "rng_seed": args.seed, "n_deep_runs": len(runs),
                 "event_seeds": event_seeds, "init_seeds": init_seeds},
        "partitions": {
            str(es): {
                "events": cluster_cache[es]["events"],
                "summaries": cluster_cache[es]["summaries"],
                "n_clusters": int(len(np.unique(cluster_cache[es]["cluster"]))),
            } for es in event_seeds if es in cluster_cache
        },
        "per_run": per_run,
        "across_partitions": across,
        "histgb_partitions": hist_rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [
        "multi-partition CLUSTER BOOTSTRAP (buffer population; unrestricted point estimates)",
        f"reps={args.reps} deep_runs={len(runs)} partitions={event_seeds}",
        "",
        "== Across partitions (hierarchical: partitions x inits x clusters) ==",
    ]
    for r in across:
        s = r["summary_buffer"]
        lines.append(
            f"{r['model']:10s} F1 {s['f1']['point']:.4f} [{s['f1']['pct_lo']:.4f},{s['f1']['pct_hi']:.4f}] | "
            f"dF1 {s['delta_f1']['point']:+.4f} [{s['delta_f1']['pct_lo']:+.4f},{s['delta_f1']['pct_hi']:+.4f}] | "
            f"AUROC {s['auroc']['point']:.4f} [{s['auroc']['pct_lo']:.4f},{s['auroc']['pct_hi']:.4f}] | "
            f"MCC {s['mcc']['point']:.4f} | above trivial {r['above_trivial_buffer']}/{r['n_partitions']}"
        )
        lines.append(f"  per-part dF1: {[round(x,4) for x in r['per_partition_buffer']['delta_f1']]}")
        lines.append(f"  per-part AUROC: {[round(x,4) for x in r['per_partition_buffer']['auroc']]}")
        lines.append(
            f"  unrestricted mean AUROC {np.mean(r['per_partition_unrestricted']['auroc']):.4f} | "
            f"mean dF1 {np.mean(r['per_partition_unrestricted']['delta_f1']):+.4f}"
        )
    if hist_rows:
        lines.append("")
        lines.append("== HistGB_5S_fixedHP across partitions ==")
        df = [r["buffer"]["delta_f1"] for r in hist_rows]
        au = [r["buffer"]["auroc"] for r in hist_rows]
        lines.append(
            f"n={len(hist_rows)} buf dF1 {np.mean(df):+.4f} AUROC {np.mean(au):.4f} | "
            f"above trivial {sum(1 for x in df if x>0)}/{len(df)} | "
            f"unr AUROC {np.mean([r['unrestricted']['auroc'] for r in hist_rows]):.4f}"
        )
    text = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text, flush=True)
    print(f"Wrote {OUT_JSON} and {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
