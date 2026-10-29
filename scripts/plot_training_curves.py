#!/usr/bin/env python3
"""Plot Part Three curves from existing language.json training records.

Run from the repository root:
    python scripts/plot_training_curves.py

The figure uses the recorded points directly, without smoothing or resampling.
Only cross-entropy is plotted; the auxiliary balancing loss is excluded.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = (
    ("dense_compute", "Dense (compute matched)", "#0072B2", "o"),
    ("moe", "Sparse MoE", "#D55E00", "s"),
    ("dense_parameter", "Dense (parameter matched)", "#009E73", "^"),
)


def load_curves(path, seed):
    """Select one run per configuration and validate its recorded values."""
    data = json.loads(path.read_text(encoding="utf-8"))
    curves = {}
    for mode, _, _, _ in CONFIGURATIONS:
        matches = [
            run for run in data["runs"]
            if run["mode"] == mode and run["seed"] == seed
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one run for {mode}, seed {seed}; found {len(matches)}."
            )
        history = matches[0]["history"]
        if not history:
            raise ValueError(f"Empty history for {mode}.")
        points = sorted(
            (record["step"], record["train_ce"], record["validation"]["ce"])
            for record in history
        )
        if len({point[0] for point in points}) != len(points):
            raise ValueError(f"Duplicate training steps for {mode}.")
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError(f"Non-finite step or cross-entropy for {mode}.")
        curves[mode] = points
    return curves


def plot_curves(curves, seed, out_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    for mode, label, color, marker in CONFIGURATIONS:
        steps, train_ce, validation_ce = zip(*curves[mode])
        for ax, values in zip(axes, (train_ce, validation_ce)):
            ax.plot(
                steps, values, label=label, color=color,
                marker=marker, markersize=5, linewidth=1.8,
            )

    steps = sorted({point[0] for points in curves.values() for point in points})
    for ax, title in zip(axes, ("(a) Training", "(b) Validation")):
        ax.set_title(title, pad=12)
        ax.set_xlabel("Optimizer update")
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.8)
        ax.set_axisbelow(True)
        if len(steps) <= 10:
            ax.set_xticks(steps)
    axes[0].set_ylabel("Cross-entropy (nats / byte)")
    fig.suptitle(f"Training and validation cross-entropy (seed {seed})", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075),
        ncol=3, frameon=False,
    )
    fig.text(
        0.5, 0.035,
        "Training: logged minibatch CE (no auxiliary term). "
        "Validation: mean CE across the two domains.",
        ha="center", fontsize=8, color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.81, bottom=0.25, wspace=0.12)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = out_dir / f"moe_training_validation_seed{seed}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "HW1_Release/results/language.json",
        help="Existing training log (default: repository HW1_Release/results/language.json).",
    )
    parser.add_argument("--seed", type=int, default=416)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "submission/figures",
        help="Output directory (default: repository submission/figures).",
    )
    args = parser.parse_args()
    try:
        curves = load_curves(args.input, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for mode, points in curves.items():
        print(f"{mode}: {len(points)} recorded points, steps {points[0][0]}–{points[-1][0]}")
    for path in plot_curves(curves, args.seed, args.out_dir):
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
