import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

# display names for the result files
APPROACH_LABELS = {
    "a1":          "A1 Raw",
    "a2":          "A2 Fixed MFCC",
    "a3":          "A3 Event-Triggered",
    "a4_high":     "A4 High Res",
    "a4_medium":   "A4 Medium Res",
    "a4_low":      "A4 Low Res",
    "a5_dim16":    "A5 Embed (16)",
    "a5_dim32":    "A5 Embed (32)",
    "a5_dim64":    "A5 Embed (64)",
    "a5_dim128":   "A5 Embed (128)",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compute_metrics(records):
    """
    Returns a dict with accuracy, macro F1, avg bytes per utterance,
    avg RTT, and transmission rate.

    Untransmitted A3 samples (transmitted=False) count as wrong predictions
    for accuracy and F1 but contribute 0 bytes to the bandwidth average.
    This gives a fair comparison — you can't get bandwidth savings for free.
    """
    true_labels, pred_labels = [], []
    bytes_list, rtt_list = [], []
    transmitted = 0

    for r in records:
        true_labels.append(r["true"])
        # None prediction (VAD skip) treated as a wrong answer "_skipped"
        pred_labels.append(r["predicted"] if r["predicted"] is not None else "_skipped")
        bytes_list.append(r["bytes"])
        if r["transmitted"]:
            transmitted += 1
            if r["rtt"] is not None:
                rtt_list.append(r["rtt"])

    total = len(records)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))

    # exclude "_skipped" from F1 since it's not a real class
    filtered = [(t, p) for t, p in zip(true_labels, pred_labels) if p != "_skipped"]
    f1_true = [t for t, _ in filtered]
    f1_pred = [p for _, p in filtered]
    macro_f1 = f1_score(f1_true, f1_pred, average="macro", zero_division=0) if filtered else 0.0

    return {
        "accuracy":          correct / total,
        "macro_f1":          macro_f1,
        "avg_bytes":         np.mean(bytes_list),
        "avg_rtt_ms":        np.mean(rtt_list) * 1000 if rtt_list else None,
        "transmission_rate": transmitted / total,
        "n_samples":         total,
    }


def print_table(results):
    header = f"{'Approach':<25} {'Accuracy':>9} {'Macro F1':>10} {'Avg Bytes':>11} {'Avg RTT (ms)':>13} {'TX Rate':>8}"
    print("\n" + header)
    print("-" * len(header))

    for tag, metrics in sorted(results.items()):
        label   = APPROACH_LABELS.get(tag, tag)
        acc     = f"{metrics['accuracy']:.4f}"
        f1      = f"{metrics['macro_f1']:.4f}"
        byt     = f"{metrics['avg_bytes']:.0f}"
        rtt     = f"{metrics['avg_rtt_ms']:.1f}" if metrics["avg_rtt_ms"] is not None else "N/A"
        tx_rate = f"{metrics['transmission_rate']:.2f}"
        print(f"{label:<25} {acc:>9} {f1:>10} {byt:>11} {rtt:>13} {tx_rate:>8}")


def plot_pareto(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- accuracy vs bandwidth ---
    fig, ax = plt.subplots(figsize=(9, 5))

    for tag, metrics in results.items():
        label = APPROACH_LABELS.get(tag, tag)
        ax.scatter(metrics["avg_bytes"], metrics["accuracy"], s=80, zorder=3)
        ax.annotate(label, (metrics["avg_bytes"], metrics["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Avg bytes per utterance")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Bandwidth — all approaches")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_bandwidth.png"), dpi=150)
    plt.close()
    print(f"Saved plots/accuracy_vs_bandwidth.png")

    # --- accuracy vs latency (only approaches that have RTT data) ---
    rtt_results = {k: v for k, v in results.items() if v["avg_rtt_ms"] is not None}
    if rtt_results:
        fig, ax = plt.subplots(figsize=(9, 5))
        for tag, metrics in rtt_results.items():
            label = APPROACH_LABELS.get(tag, tag)
            ax.scatter(metrics["avg_rtt_ms"], metrics["accuracy"], s=80, zorder=3)
            ax.annotate(label, (metrics["avg_rtt_ms"], metrics["accuracy"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        ax.set_xlabel("Avg round-trip latency (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Latency — all approaches")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_latency.png"), dpi=150)
        plt.close()
        print(f"Saved plots/accuracy_vs_latency.png")

    # --- A4 and A5 Pareto curves (bandwidth sweep) ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # A4 resolution sweep
    a4_tags = ["a4_low", "a4_medium", "a4_high"]
    a4_tags = [t for t in a4_tags if t in results]
    if a4_tags:
        a4_bytes = [results[t]["avg_bytes"]  for t in a4_tags]
        a4_acc   = [results[t]["accuracy"]   for t in a4_tags]
        ax.plot(a4_bytes, a4_acc, "o-", label="A4 Dynamic MFCC")

    # A5 embedding dim sweep
    a5_tags = ["a5_dim16", "a5_dim32", "a5_dim64", "a5_dim128"]
    a5_tags = [t for t in a5_tags if t in results]
    if a5_tags:
        a5_bytes = [results[t]["avg_bytes"] for t in a5_tags]
        a5_acc   = [results[t]["accuracy"]  for t in a5_tags]
        ax.plot(a5_bytes, a5_acc, "s-", label="A5 Learned Embedding")

    # fixed approaches as reference points
    for tag in ["a1", "a2", "a3"]:
        if tag in results:
            label = APPROACH_LABELS[tag]
            ax.scatter(results[tag]["avg_bytes"], results[tag]["accuracy"],
                       marker="^", s=100, zorder=3, label=label)

    ax.set_xlabel("Avg bytes per utterance")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Curves — Accuracy vs Bandwidth")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pareto_curves.png"), dpi=150)
    plt.close()
    print(f"Saved plots/pareto_curves.png")


def main():
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    if not json_files:
        print(f"No result files found in {RESULTS_DIR}/")
        print("Run device.py for each approach first.")
        return

    results = {}
    for path in json_files:
        tag = os.path.splitext(os.path.basename(path))[0]
        records = load_results(path)
        results[tag] = compute_metrics(records)
        print(f"Loaded {tag}: {len(records)} samples")

    print_table(results)
    plot_pareto(results)


if __name__ == "__main__":
    main()
