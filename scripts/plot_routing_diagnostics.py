#!/usr/bin/env python3
"""Plot Part Three expert usage and the last-block routing diagnostic.

Run from the repository root:
    python scripts/plot_routing_diagnostics.py

Reads existing results only. The three shuffle trials are shown individually;
their mean and observed range are not confidence intervals. No diagnostic PPL
is calculated because shuffled routes depend on other sequences in the batch.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_LABELS = {
    "shakespeare": "Text (Shakespeare)",
    "python": "Code (Python)",
}


def load_diagnostic(path, seed):
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [r for r in data["routing_control"] if r["seed"] == seed]
    if len(records) != 1:
        raise ValueError(f"Expected one routing diagnostic for seed {seed}.")
    record = records[0]
    normal = record["normal_test"]
    domains = data["metadata"]["domains"].split(",")
    experts = data["metadata"]["experts"]
    usage = np.asarray(normal["route_fraction"], dtype=float)
    if len(domains) != 2 or usage.shape != (2, 2, experts):
        raise ValueError("Expected two domains, two MoE blocks, and E experts.")
    if not np.isfinite(usage).all() or (usage < 0).any():
        raise ValueError("Expert usage must be finite and nonnegative.")
    if not np.allclose(usage.sum(axis=-1), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("Each domain/block expert-usage distribution must sum to one.")

    trials = record["position_wise_trials"]
    if len(trials) != 3:
        raise ValueError("Expected the three prescribed routing-shuffle trials.")
    if [trial["routing_seed"] for trial in trials] != record["routing_seeds"]:
        raise ValueError("Trial seeds disagree with the recorded routing seeds.")
    for trial in trials:
        diagnostic = trial["diagnostic"]
        if diagnostic["route_fraction"] != normal["route_fraction"]:
            raise ValueError("Expert usage changed under the routing perturbation.")
        if diagnostic["evaluated_targets_per_domain"] != normal["evaluated_targets_per_domain"]:
            raise ValueError("Normal and perturbed evaluations use different target counts.")
    ces = [normal["ce"]] + [trial["diagnostic"]["ce"] for trial in trials]
    if not all(math.isfinite(ce) and ce >= 0 for ce in ces):
        raise ValueError("Cross-entropies must be finite and nonnegative.")
    return domains, usage, trials, ces


def plot_diagnostic(domains, usage, trials, ces, seed, out_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.4, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1))
    usage_axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    ce_ax = fig.add_subplot(grid[1, :])
    experts = usage.shape[-1]
    positions = np.arange(experts)
    colors = ("#0072B2", "#D55E00")
    ceiling = math.ceil(max(float(usage.max()) * 1.2, 1.3 / experts) * 20) / 20

    for block, ax in enumerate(usage_axes):
        for domain, color in enumerate(colors):
            ax.bar(
                positions + (-0.2 if domain == 0 else 0.2), usage[domain, block],
                width=0.38, color=color,
                label=DOMAIN_LABELS.get(domains[domain], domains[domain]),
                zorder=3,
            )
        ax.axhline(
            1 / experts, color="#555555", linestyle="--", linewidth=1,
            label=f"Uniform share (1/{experts})", zorder=4,
        )
        ax.set_title(f"({'ab'[block]}) Block {block + 1}: expert usage", loc="left", pad=10)
        ax.set_xlabel("Expert ID (0-based)")
        ax.set_ylabel("Share of selected token–expert pairs")
        ax.set_xticks(positions)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlim(-0.7, experts - 0.3)
        ax.set_ylim(0, ceiling)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.8)
        ax.set_axisbelow(True)

    labels = ["Normal routes"] + [f"Shuffle {trial['routing_seed']}" for trial in trials]
    ce_ax.barh(
        np.arange(len(ces)), ces, height=0.58,
        color=["#009E73"] + ["#8064A2"] * len(trials), zorder=3,
    )
    ce_ax.set_yticks(np.arange(len(ces)), labels)
    ce_ax.invert_yaxis()
    ce_ax.set_xlim(0, max(ces) * 1.13)
    ce_ax.set_xlabel("Cross-entropy (nats / byte; lower is better)")
    ce_ax.set_title("(c) Last-block routing perturbation", loc="left", pad=20)
    ce_ax.grid(axis="x", color="#E2E5E9", linewidth=0.8)
    ce_ax.set_axisbelow(True)
    for row, value in enumerate(ces):
        ce_ax.text(value + max(ces) * 0.012, row, f"{value:.4f}", va="center", fontsize=10)

    mean = float(np.mean(ces[1:]))
    ce_ax.text(
        1, 1.035,
        f"Shuffle mean: {mean:.4f}  |  range: {min(ces[1:]):.4f}–{max(ces[1:]):.4f}"
        f"  |  ΔCE: +{mean - ces[0]:.4f}",
        transform=ce_ax.transAxes, ha="right", va="bottom", fontsize=9,
    )
    fig.suptitle(f"MoE expert usage and routing diagnostic (model seed {seed})", fontsize=14, y=0.99)
    handles, legend_labels = usage_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952),
        ncol=3, frameon=False,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.09, hspace=0.65, wspace=0.26)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = out_dir / f"moe_expert_usage_routing_seed{seed}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "HW1_Release/results/language.json",
    )
    parser.add_argument("--seed", type=int, default=416)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "submission/figures")
    args = parser.parse_args()
    try:
        domains, usage, trials, ces = load_diagnostic(args.input, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Normal CE: {ces[0]:.6f}; mean shuffle diagnostic CE: {np.mean(ces[1:]):.6f}")
    for path in plot_diagnostic(domains, usage, trials, ces, args.seed, args.out_dir):
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
