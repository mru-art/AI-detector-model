"""Reads results/<method>/metrics.json for every method and makes one comparison table + plot."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_common import config as cfg

METHODS = ["naive", "replay", "ewc", "kd"]


def compare_results():
    results = {}
    for m in METHODS:
        path = os.path.join(cfg.RESULTS_DIR, m, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
    if not results:
        print("No results found yet.")
        return

    lines = [f"{'method':8s} {'final avg acc':>14s} {'macro-F1':>9s} {'real-vs-AI':>11s} {'forgetting':>11s} {'backward transfer':>18s}"]
    for m, r in results.items():
        s = r["summary"]
        lines.append(f"{m:8s} {s['final_avg_acc'] * 100:13.1f}% {s['final_macro_f1'] * 100:8.1f}% "
                     f"{s['final_avg_real_vs_ai'] * 100:10.1f}% {s['final_forgetting'] * 100:10.1f}  {s['final_backward_transfer'] * 100:17.1f}")
    lines.append("\n(final avg acc / macro-F1 / real-vs-AI are measured after the last experience; "
                 "forgetting and backward transfer are in percentage points)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(cfg.RESULTS_DIR, "comparison.txt"), "w") as f:
        f.write(text + "\n")

    x = list(range(1, len(cfg.GENERATORS) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m, r in results.items():
        axes[0].plot(x, [a * 100 for a in r["avg_acc"]], marker="o", label=m)
        axes[1].plot(x[1:], [f * 100 for f in r["forgetting"][1:]], marker="o", label=m)
    axes[0].set_title("Average 6-way accuracy over all classes seen so far")
    axes[0].set_ylabel("accuracy (%)")
    axes[1].set_title("Forgetting of old generators (lower is better)")
    axes[1].set_ylabel("percentage points")
    for ax in axes:
        ax.set_xlabel("after experience")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()
    print(f"Saved comparison.txt and comparison.png to {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    compare_results()
