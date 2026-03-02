#!/usr/bin/env python3
"""
Analyze & visualize the Myanmar ASR dataset.
Produces PNG plots + a JSON summary saved to /workspace/analysis/.
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

OUT_DIR = "/workspace/analysis"
DATA_DIR = "/workspace/data/myanmar_asr"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"train": "#2196F3", "validation": "#FF9800", "test": "#4CAF50"}


def main():
    from datasets import load_from_disk

    print("Loading dataset...")
    ds = load_from_disk(DATA_DIR)
    print(f"  Splits: {list(ds.keys())}")
    for s in ds:
        print(f"  {s}: {len(ds[s]):,} samples, cols={ds[s].column_names}")

    stats = {}

    # ── Gather per-split statistics ───────────────────────────
    for split in ds:
        print(f"\nAnalyzing {split}...")
        data = ds[split]
        n = len(data)

        durations = []
        text_lens = []
        sources = []
        rms_vals = []

        for i in range(n):
            row = data[i]
            audio = row["audio"]
            arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            dur = len(arr) / sr
            durations.append(dur)

            txt = row.get("sentence", "") or ""
            text_lens.append(len(txt))
            sources.append(row.get("source", "unknown"))
            rms_vals.append(float(np.sqrt(np.mean(arr ** 2))))

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{n}")

        durations = np.array(durations)
        text_lens = np.array(text_lens)
        rms_vals = np.array(rms_vals)
        source_counts = Counter(sources)

        stats[split] = {
            "samples": n,
            "total_hours": round(float(durations.sum()) / 3600, 2),
            "duration_mean": round(float(durations.mean()), 2),
            "duration_std": round(float(durations.std()), 2),
            "duration_min": round(float(durations.min()), 2),
            "duration_max": round(float(durations.max()), 2),
            "duration_median": round(float(np.median(durations)), 2),
            "text_len_mean": round(float(text_lens.mean()), 1),
            "text_len_std": round(float(text_lens.std()), 1),
            "text_len_min": int(text_lens.min()),
            "text_len_max": int(text_lens.max()),
            "rms_mean": round(float(rms_vals.mean()), 5),
            "rms_min": round(float(rms_vals.min()), 6),
            "sources": dict(source_counts),
            "durations": durations,
            "text_lens": text_lens,
            "rms_vals": rms_vals,
        }

    # ── Save JSON summary (no numpy arrays) ───────────────────
    json_stats = {}
    for s in stats:
        json_stats[s] = {k: v for k, v in stats[s].items()
                         if not isinstance(v, np.ndarray)}
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nSaved stats → {OUT_DIR}/dataset_stats.json")

    # ── Plot 1: Duration Distribution ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split in zip(axes, ["train", "validation", "test"]):
        if split not in stats:
            continue
        d = stats[split]["durations"]
        ax.hist(d, bins=60, color=COLORS[split], alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(d), color="red", linestyle="--", lw=1.5,
                   label=f"mean={np.mean(d):.1f}s")
        ax.axvline(np.median(d), color="orange", linestyle=":", lw=1.5,
                   label=f"median={np.median(d):.1f}s")
        ax.set_title(f"{split} ({len(d):,} samples, {d.sum()/3600:.1f}h)",
                     fontweight="bold")
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    fig.suptitle("Audio Duration Distribution per Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_duration_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Plot 1: duration distribution")

    # ── Plot 2: Text Length Distribution ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split in zip(axes, ["train", "validation", "test"]):
        if split not in stats:
            continue
        t = stats[split]["text_lens"]
        ax.hist(t, bins=50, color=COLORS[split], alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(t), color="red", linestyle="--", lw=1.5,
                   label=f"mean={np.mean(t):.0f} chars")
        ax.set_title(f"{split} (mean={np.mean(t):.0f}, max={t.max()})",
                     fontweight="bold")
        ax.set_xlabel("Text Length (characters)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    fig.suptitle("Transcription Length Distribution per Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_text_length_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Plot 2: text length distribution")

    # ── Plot 3: Source Composition ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = sns.color_palette("Set2", 10)
    for ax, split in zip(axes, ["train", "validation", "test"]):
        if split not in stats:
            continue
        src = stats[split]["sources"]
        labels = sorted(src.keys())
        values = [src[l] for l in labels]
        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct="%1.1f%%",
            colors=palette[:len(labels)], startangle=90,
            textprops={"fontsize": 9}
        )
        ax.legend(labels, loc="lower left", fontsize=8)
        ax.set_title(f"{split} ({sum(values):,} samples)", fontweight="bold")
    fig.suptitle("Data Source Composition per Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_source_composition.png"), dpi=150)
    plt.close(fig)
    print("  Plot 3: source composition")

    # ── Plot 4: Duration vs Text Length scatter ───────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for split in ["train", "validation", "test"]:
        if split not in stats:
            continue
        d = stats[split]["durations"]
        t = stats[split]["text_lens"]
        ax.scatter(d, t, alpha=0.2, s=8, color=COLORS[split], label=split)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Text Length (characters)")
    ax.set_title("Duration vs Transcription Length", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_duration_vs_textlen.png"), dpi=150)
    plt.close(fig)
    print("  Plot 4: duration vs text length")

    # ── Plot 5: RMS Energy Distribution ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for split in ["train", "validation", "test"]:
        if split not in stats:
            continue
        r = stats[split]["rms_vals"]
        ax.hist(r, bins=80, alpha=0.6, color=COLORS[split], label=split,
                edgecolor="white")
    ax.set_xlabel("RMS Energy")
    ax.set_ylabel("Count")
    ax.set_title("Audio RMS Energy Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_rms_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Plot 5: RMS energy distribution")

    # ── Plot 6: Summary Dashboard ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 6a: Samples per split bar
    ax = axes[0, 0]
    splits = list(json_stats.keys())
    counts = [json_stats[s]["samples"] for s in splits]
    bars = ax.bar(splits, counts, color=[COLORS.get(s, "#999") for s in splits],
                  edgecolor="white", linewidth=1.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{c:,}", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Samples per Split", fontweight="bold")
    ax.set_ylabel("Count")

    # 6b: Hours per split bar
    ax = axes[0, 1]
    hours = [json_stats[s]["total_hours"] for s in splits]
    bars = ax.bar(splits, hours, color=[COLORS.get(s, "#999") for s in splits],
                  edgecolor="white", linewidth=1.5)
    for bar, h in zip(bars, hours):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{h:.1f}h", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Total Hours per Split", fontweight="bold")
    ax.set_ylabel("Hours")

    # 6c: Duration box plot
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    for split in splits:
        if split in stats:
            box_data.append(stats[split]["durations"])
            box_labels.append(split)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showfliers=False)
    for patch, split in zip(bp["boxes"], box_labels):
        patch.set_facecolor(COLORS.get(split, "#999"))
        patch.set_alpha(0.7)
    ax.set_title("Duration Distribution (no outliers)", fontweight="bold")
    ax.set_ylabel("Seconds")

    # 6d: Source breakdown (train only)
    ax = axes[1, 1]
    if "train" in stats:
        src = stats["train"]["sources"]
        labels = sorted(src.keys())
        values = [src[l] for l in labels]
        bars = ax.barh(labels, values, color=palette[:len(labels)],
                       edgecolor="white", linewidth=1)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                    f"{v:,}", va="center", fontsize=10)
        ax.set_title("Train Source Breakdown", fontweight="bold")
        ax.set_xlabel("Samples")

    fig.suptitle("Myanmar ASR Dataset — Analysis Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_summary_dashboard.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Plot 6: summary dashboard")

    # ── Print final summary ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET ANALYSIS COMPLETE")
    print(f"{'='*60}")
    total_samples = sum(json_stats[s]["samples"] for s in json_stats)
    total_hours = sum(json_stats[s]["total_hours"] for s in json_stats)
    print(f"  Total: {total_samples:,} samples | {total_hours:.1f}h")
    for s in json_stats:
        st = json_stats[s]
        print(f"  {s:12s}: {st['samples']:>6,} samples | {st['total_hours']:>5.1f}h "
              f"| dur {st['duration_mean']:.1f}±{st['duration_std']:.1f}s "
              f"| txt {st['text_len_mean']:.0f}±{st['text_len_std']:.0f} chars")
    print(f"\n  Plots saved to: {OUT_DIR}/")
    print(f"  Files: {', '.join(sorted(os.listdir(OUT_DIR)))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
