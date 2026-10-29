#!/usr/bin/env python3
"""Evaluate fixed last-block Top-2 routes with interpolated gate weights.

From the repository root:
    python scripts/explore_gate_weights.py --device cuda
    python scripts/explore_gate_weights.py --plot-only

alpha=1 retains learned gates; alpha=0 assigns 1/2 to both selected experts.
Only the final FFN's route return value is temporarily wrapped. Model weights,
expert IDs, earlier blocks, and the supplied full-test evaluation are unchanged.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch
from torch.torch_version import TorchVersion

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "HW1_Release/MoE"))
from train import Corpus, Transformer, evaluate_full


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RouteTrace:
    """Fingerprint every selected expert ID in evaluation order."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.digest = hashlib.sha256()
        self.tokens = self.batches = 0
        self.original_top1_sum = self.used_top1_sum = 0.0
        self.max_gate_sum_error = 0.0

    def update(self, ids, original_gates, used_gates):
        self.digest.update(ids.detach().cpu().contiguous().numpy().tobytes())
        original = original_gates.detach().cpu().double().numpy()
        used = used_gates.detach().cpu().double().numpy()
        self.original_top1_sum += float(original[:, 0].sum())
        self.used_top1_sum += float(used[:, 0].sum())
        self.max_gate_sum_error = max(
            self.max_gate_sum_error, float(np.abs(used.sum(axis=-1) - 1).max())
        )
        if self.alpha == 0 and not np.all(used == 0.5):
            raise RuntimeError("The equal-weight condition is not exactly 1/2, 1/2.")
        if self.alpha == 1 and not np.array_equal(original, used):
            raise RuntimeError("The alpha=1 condition changed the learned gates.")
        self.tokens += len(original)
        self.batches += 1
        if self.batches % 100 == 0:
            print(f"alpha={self.alpha:g}: evaluated {self.tokens:,} token positions", flush=True)

    def summary(self):
        if self.max_gate_sum_error > 1e-6:
            raise RuntimeError("Interpolated gate weights do not sum to one.")
        return {
            "expert_ids_sha256": self.digest.hexdigest(),
            "token_positions": self.tokens,
            "selected_token_expert_pairs": 2 * self.tokens,
            "original_mean_top1_weight": self.original_top1_sum / self.tokens,
            "used_mean_top1_weight": self.used_top1_sum / self.tokens,
            "max_gate_sum_error": self.max_gate_sum_error,
        }


@contextmanager
def interpolate_last_route(model, alpha, trace=None):
    layer = model.blocks[-1].ffn
    if not layer.moe or layer.topk != 2:
        raise ValueError("This investigation requires a last-block Top-2 MoE.")
    original_route = layer.route

    def route(flat):
        all_probs, gates, ids = original_route(flat)
        adjusted = alpha * gates + (1 - alpha) / 2
        if trace is not None:
            trace.update(ids, gates, adjusted)
        return all_probs, adjusted, ids

    layer.route = route
    try:
        yield
    finally:
        layer.route = original_route


def evaluate_conditions(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Metadata may contain this known string subclass; keep weights-only loading.
    torch.serialization.add_safe_globals([TorchVersion])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = checkpoint["args"]
    if sha256(args.corpus) != configuration["corpus_sha256"]:
        raise ValueError("The corpus differs from the checkpoint's recorded corpus.")
    model = Transformer(
        configuration["context"], configuration["d"], configuration["width"],
        True, experts=configuration["experts"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch = configuration["batch"]
    domains = configuration["domains"].split(",")
    corpus = Corpus(args.corpus, configuration["context"], "transformer", domains, device)

    # A small identity check precedes the full, prescribed evaluation.
    probe = torch.arange(2 * configuration["context"], device=device).reshape(2, -1) % 256
    with torch.no_grad():
        expected, _ = model(probe)
        with interpolate_last_route(model, 1.0):
            actual, _ = model(probe)
    identity_error = float((actual - expected).abs().max().item())
    if identity_error != 0:
        raise RuntimeError("The alpha=1 wrapper does not reproduce the original model.")

    runs = []
    for alpha in (1.0, 0.5, 0.0):
        print(f"Starting full test evaluation: alpha={alpha:g}, device={device}", flush=True)
        trace = RouteTrace(alpha)
        with interpolate_last_route(model, alpha, trace):
            metrics = evaluate_full(model, corpus, "test", batch)
        summary = trace.summary()
        if summary["token_positions"] != sum(metrics["evaluated_targets_per_domain"]):
            raise RuntimeError("Route tracing did not cover exactly the evaluated targets.")
        if runs:
            baseline = runs[0]
            if summary["expert_ids_sha256"] != baseline["route_trace"]["expert_ids_sha256"]:
                raise RuntimeError("Selected expert IDs changed between weight conditions.")
            for key in ("route_fraction", "evaluated_targets_per_domain"):
                if metrics[key] != baseline["test"][key]:
                    raise RuntimeError(f"The controlled quantity {key} changed.")
        runs.append({"alpha": alpha, "test": metrics, "route_trace": summary})
        print(json.dumps({
            "alpha": alpha, "domain_ce": metrics["domain_ce"],
            "ce": metrics["ce"], "perplexity": metrics["perplexity"],
        }), flush=True)

    # Do not modify the original language.json or the checkpoint.
    return {
        "metadata": {
            "question": "Do learned last-block gate weights help beyond fixed expert selection?",
            "intervention": "g_alpha = alpha * g_learned + (1-alpha) / 2; last block only",
            "alphas_fixed_before_evaluation": [1.0, 0.5, 0.0],
            "model_seed": args.seed, "retrained": False, "split": "test",
            "domains": domains, "context": configuration["context"], "batch": batch,
            "torch": str(torch.__version__), "python": platform.python_version(),
            "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
            "dtype": str(next(model.parameters()).dtype),
            "checkpoint_sha256": sha256(args.checkpoint),
            "corpus_sha256": sha256(args.corpus), "script_sha256": sha256(Path(__file__)),
            "original_training_device": configuration["device"],
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "controls": {
            "alpha1_identity_max_logit_error": identity_error,
            "all_token_expert_ids_identical": True,
            "all_layers_expert_counts_identical": True,
            "evaluation_target_counts_identical": True,
            "all_parameters_frozen": True,
            "earlier_blocks_unchanged": True,
            "independent_sequence_evaluation": True,
        },
        "runs": runs,
    }


def plot_results(data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    runs = sorted(data["runs"], key=lambda run: run["alpha"])
    alpha = [run["alpha"] for run in runs]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for index, label, color, marker in (
        (0, "Text (Shakespeare)", "#0072B2", "o"),
        (1, "Code (Python)", "#D55E00", "s"),
        (None, "Equal-domain mean", "#009E73", "^"),
    ):
        values = [run["test"]["ce"] if index is None else run["test"]["domain_ce"][index] for run in runs]
        ax.plot(alpha, values, color=color, marker=marker, label=label, linewidth=1.8)
        for x, value in zip(alpha, values):
            ax.annotate(f"{value:.4f}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks([0, 0.5, 1], ["0\nEqual weights", "0.5\nHalf interpolation", "1\nLearned weights"])
    ax.set_xlabel("Learned-weight fraction α")
    ax.set_ylabel("Test cross-entropy (nats / byte)")
    ax.set_title("Fixed last-block expert IDs: gate-weight intervention", pad=14)
    ax.grid(axis="y", color="#E2E5E9")
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.18)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.subplots_adjust(bottom=0.30, left=0.12, right=0.98, top=0.88)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        path = out_dir / f"moe_gate_weights_seed{data['metadata']['model_seed']}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=416)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "HW1_Release/data/corpus.npz")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "submission/figures")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the saved results without reevaluating.")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or REPO_ROOT / f"HW1_Release/results/moe_seed{args.seed}.pt"
    args.results = args.results or REPO_ROOT / f"HW1_Release/results/gate_weights_seed{args.seed}.json"
    if args.plot_only:
        data = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        data = evaluate_conditions(args)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.results}", flush=True)
    for path in plot_results(data, args.out_dir):
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
