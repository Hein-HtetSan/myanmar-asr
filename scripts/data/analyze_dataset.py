import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from datasets import load_from_disk
import json, os

os.makedirs("viz", exist_ok=True)

# ── Load ALL splits ───────────────────────────────────────────
# Use augmented dataset if available, otherwise clean
import sys
dataset_path = "combined/myanmar_asr_augmented"
if not os.path.exists(dataset_path):
    dataset_path = "combined/myanmar_asr_50h_clean"
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]

print(f"Loading dataset from {dataset_path}...")
ds = load_from_disk(dataset_path)

all_rows = []
for split_name in ds:
    print(f"  Processing {split_name} ({len(ds[split_name]):,} samples)...")
    for example in ds[split_name]:
        audio = example["audio"]
        dur = len(audio["array"]) / audio["sampling_rate"]
        all_rows.append({
            "split": split_name,
            "source": example["source"],
            "speaker_id": example["speaker_id"],
            "duration": dur,
            "text_len": len(example["sentence"]),
            "sentence": example["sentence"],
        })

df = pd.DataFrame(all_rows)
df_train = df[df["split"] == "train"]

# ── COLORS ────────────────────────────────────────────────────
SRC_COLORS = {"fleurs": "#5B8DEF", "openslr80": "#F5A623", "yodas_my": "#7ED321"}
SPLIT_COLORS = {"train": "#5B8DEF", "validation": "#F5A623", "test": "#E74C3C"}

# ════════════════════════════════════════════════════════════════
#  IMAGE 1: Overview Dashboard  (4 panels)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Myanmar ASR — Dataset Overview Dashboard", fontsize=18, fontweight="bold", y=0.98)

# 1a. Duration distribution (all data)
axes[0,0].hist(df["duration"], bins=80, color="#5B8DEF", edgecolor="white", alpha=0.85)
axes[0,0].axvline(df["duration"].mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {df['duration'].mean():.1f}s")
axes[0,0].axvline(df["duration"].median(), color="orange", linestyle=":", linewidth=2,
                   label=f"Median: {df['duration'].median():.1f}s")
axes[0,0].set_title("Audio Duration Distribution (All Splits)", fontsize=13, fontweight="bold")
axes[0,0].set_xlabel("Duration (seconds)")
axes[0,0].set_ylabel("Count")
axes[0,0].legend(fontsize=10)

# 1b. Source pie chart
src_counts = df["source"].value_counts()
colors_pie = [SRC_COLORS.get(s, "#999") for s in src_counts.index]
wedges, texts, autotexts = axes[0,1].pie(
    src_counts.values, labels=src_counts.index, autopct="%1.1f%%",
    colors=colors_pie, startangle=90, textprops={"fontsize": 11})
for t in autotexts:
    t.set_fontweight("bold")
axes[0,1].set_title("Samples by Dataset Source", fontsize=13, fontweight="bold")

# 1c. Text length histogram
axes[1,0].hist(df["text_len"], bins=80, color="#7ED321", edgecolor="white", alpha=0.85)
axes[1,0].axvline(df["text_len"].mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {df['text_len'].mean():.0f} chars")
axes[1,0].set_title("Transcript Length Distribution", fontsize=13, fontweight="bold")
axes[1,0].set_xlabel("Characters")
axes[1,0].set_ylabel("Count")
axes[1,0].legend(fontsize=10)

# 1d. Hours by source (stacked by split)
hours_pivot = df.groupby(["source", "split"])["duration"].sum().unstack(fill_value=0) / 3600
hours_pivot = hours_pivot.reindex(columns=["train", "validation", "test"], fill_value=0)
hours_pivot.plot(kind="bar", stacked=True, ax=axes[1,1],
                 color=[SPLIT_COLORS[c] for c in hours_pivot.columns], edgecolor="white")
axes[1,1].set_title("Hours by Source & Split", fontsize=13, fontweight="bold")
axes[1,1].set_ylabel("Hours")
axes[1,1].set_xlabel("")
axes[1,1].tick_params(axis="x", rotation=0)
# Add total labels on bars
for i, src in enumerate(hours_pivot.index):
    total = hours_pivot.loc[src].sum()
    axes[1,1].text(i, total + 0.1, f"{total:.1f}h", ha="center", fontweight="bold", fontsize=11)
axes[1,1].legend(title="Split", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("viz/01_overview_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved viz/01_overview_dashboard.png")

# ════════════════════════════════════════════════════════════════
#  IMAGE 2: Hours & Splits Summary
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Myanmar ASR — Hours & Splits Breakdown", fontsize=18, fontweight="bold", y=1.02)

# 2a. Total hours per split
split_hours = df.groupby("split")["duration"].sum() / 3600
split_hours = split_hours.reindex(["train", "validation", "test"], fill_value=0)
bars = axes[0].bar(split_hours.index, split_hours.values,
                    color=[SPLIT_COLORS[s] for s in split_hours.index], edgecolor="white", width=0.6)
for bar, val in zip(bars, split_hours.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f"{val:.2f}h\n({val*60:.0f} min)", ha="center", fontweight="bold", fontsize=11)
axes[0].set_title("Total Hours per Split", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Hours")

# 2b. Sample count per split
split_counts = df.groupby("split").size()
split_counts = split_counts.reindex(["train", "validation", "test"], fill_value=0)
bars = axes[1].bar(split_counts.index, split_counts.values,
                    color=[SPLIT_COLORS[s] for s in split_counts.index], edgecolor="white", width=0.6)
for bar, val in zip(bars, split_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 50,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=12)
axes[1].set_title("Samples per Split", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count")

# 2c. Big total hours display
total_hours = df["duration"].sum() / 3600
total_samples = len(df)
ax = axes[2]
ax.axis("off")
ax.text(0.5, 0.65, f"{total_hours:.1f}", fontsize=72, fontweight="bold",
        ha="center", va="center", color="#5B8DEF",
        transform=ax.transAxes)
ax.text(0.5, 0.42, "TOTAL HOURS", fontsize=18, fontweight="bold",
        ha="center", va="center", color="#333",
        transform=ax.transAxes)
ax.text(0.5, 0.25, f"{total_samples:,} total samples", fontsize=14,
        ha="center", va="center", color="#666",
        transform=ax.transAxes)
ax.text(0.5, 0.12, f"Mean duration: {df['duration'].mean():.1f}s  |  "
        f"Min: {df['duration'].min():.1f}s  |  Max: {df['duration'].max():.1f}s",
        fontsize=11, ha="center", va="center", color="#888",
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig("viz/02_hours_splits_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved viz/02_hours_splits_summary.png")

# ════════════════════════════════════════════════════════════════
#  IMAGE 3: Per-Source Detailed Analysis
# ════════════════════════════════════════════════════════════════
sources = sorted(df["source"].unique())
fig, axes = plt.subplots(len(sources), 3, figsize=(18, 5 * len(sources)))
fig.suptitle("Myanmar ASR — Per-Source Detailed Analysis", fontsize=18, fontweight="bold", y=1.01)

for row_idx, src in enumerate(sources):
    dfs = df[df["source"] == src]
    color = SRC_COLORS.get(src, "#999")

    # Duration histogram
    axes[row_idx, 0].hist(dfs["duration"], bins=50, color=color, edgecolor="white", alpha=0.85)
    axes[row_idx, 0].axvline(dfs["duration"].mean(), color="red", linestyle="--",
                              label=f"Mean: {dfs['duration'].mean():.1f}s")
    axes[row_idx, 0].set_title(f"{src} — Duration Distribution", fontweight="bold")
    axes[row_idx, 0].set_xlabel("Duration (s)")
    axes[row_idx, 0].legend(fontsize=9)

    # Text length histogram
    axes[row_idx, 1].hist(dfs["text_len"], bins=50, color=color, edgecolor="white", alpha=0.85)
    axes[row_idx, 1].axvline(dfs["text_len"].mean(), color="red", linestyle="--",
                              label=f"Mean: {dfs['text_len'].mean():.0f} chars")
    axes[row_idx, 1].set_title(f"{src} — Transcript Length", fontweight="bold")
    axes[row_idx, 1].set_xlabel("Characters")
    axes[row_idx, 1].legend(fontsize=9)

    # Stats box
    ax = axes[row_idx, 2]
    ax.axis("off")
    stats_text = (
        f"Source: {src}\n"
        f"─────────────────────\n"
        f"Samples:     {len(dfs):,}\n"
        f"Total hours: {dfs['duration'].sum()/3600:.2f}h\n"
        f"Mean dur:    {dfs['duration'].mean():.1f}s\n"
        f"Std dur:     {dfs['duration'].std():.1f}s\n"
        f"Min dur:     {dfs['duration'].min():.1f}s\n"
        f"Max dur:     {dfs['duration'].max():.1f}s\n"
        f"─────────────────────\n"
        f"Mean text:   {dfs['text_len'].mean():.0f} chars\n"
        f"Speakers:    {dfs['speaker_id'].nunique()}\n"
    )
    ax.text(0.1, 0.95, stats_text, fontsize=12, fontfamily="monospace",
            verticalalignment="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.15))

plt.tight_layout()
plt.savefig("viz/03_per_source_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved viz/03_per_source_analysis.png")

# ════════════════════════════════════════════════════════════════
#  IMAGE 4: Duration vs Text Length Scatter + Boxplots
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Myanmar ASR — Duration vs. Text Length Analysis", fontsize=18, fontweight="bold", y=1.02)

# 4a. Scatter: duration vs text_len colored by source
for src in sources:
    dfs = df[df["source"] == src]
    axes[0].scatter(dfs["duration"], dfs["text_len"], alpha=0.3, s=10,
                     color=SRC_COLORS.get(src, "#999"), label=src)
axes[0].set_xlabel("Duration (seconds)")
axes[0].set_ylabel("Transcript Length (chars)")
axes[0].set_title("Duration vs Transcript Length", fontweight="bold")
axes[0].legend(fontsize=10)

# 4b. Boxplot: duration by source
box_data = [df[df["source"] == s]["duration"].values for s in sources]
bp = axes[1].boxplot(box_data, labels=sources, patch_artist=True, showfliers=False)
for patch, src in zip(bp["boxes"], sources):
    patch.set_facecolor(SRC_COLORS.get(src, "#999"))
    patch.set_alpha(0.7)
axes[1].set_ylabel("Duration (seconds)")
axes[1].set_title("Duration by Source (Box Plot)", fontweight="bold")

# 4c. Boxplot: text length by source
box_data_t = [df[df["source"] == s]["text_len"].values for s in sources]
bp2 = axes[2].boxplot(box_data_t, labels=sources, patch_artist=True, showfliers=False)
for patch, src in zip(bp2["boxes"], sources):
    patch.set_facecolor(SRC_COLORS.get(src, "#999"))
    patch.set_alpha(0.7)
axes[2].set_ylabel("Transcript Length (chars)")
axes[2].set_title("Transcript Length by Source (Box Plot)", fontweight="bold")

plt.tight_layout()
plt.savefig("viz/04_duration_vs_text.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved viz/04_duration_vs_text.png")

# ════════════════════════════════════════════════════════════════
#  IMAGE 5: Cumulative Duration & Coverage
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Myanmar ASR — Cumulative Duration & Coverage", fontsize=18, fontweight="bold", y=1.02)

# 5a. Cumulative hours curve (sorted by duration)
sorted_durs = np.sort(df_train["duration"].values)
cumulative_hours = np.cumsum(sorted_durs) / 3600
axes[0].plot(range(len(cumulative_hours)), cumulative_hours, color="#5B8DEF", linewidth=2)
axes[0].fill_between(range(len(cumulative_hours)), cumulative_hours, alpha=0.2, color="#5B8DEF")
axes[0].set_xlabel("Sample Index (sorted by duration)")
axes[0].set_ylabel("Cumulative Hours")
axes[0].set_title("Cumulative Training Hours", fontweight="bold")
axes[0].axhline(y=cumulative_hours[-1], color="red", linestyle="--", alpha=0.5,
                 label=f"Total: {cumulative_hours[-1]:.1f}h")
axes[0].legend(fontsize=11)

# 5b. Duration buckets bar chart
buckets = [(0, 2, "0-2s"), (2, 5, "2-5s"), (5, 10, "5-10s"),
           (10, 15, "10-15s"), (15, 20, "15-20s"), (20, 30, "20-30s")]
bucket_counts = []
bucket_hours = []
bucket_labels = []
for lo, hi, label in buckets:
    mask = (df_train["duration"] >= lo) & (df_train["duration"] < hi)
    bucket_counts.append(mask.sum())
    bucket_hours.append(df_train.loc[mask, "duration"].sum() / 3600)
    bucket_labels.append(label)

x = np.arange(len(bucket_labels))
width = 0.35
bars1 = axes[1].bar(x - width/2, bucket_counts, width, color="#5B8DEF", label="Samples", alpha=0.85)
ax2 = axes[1].twinx()
bars2 = ax2.bar(x + width/2, bucket_hours, width, color="#F5A623", label="Hours", alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bucket_labels)
axes[1].set_ylabel("Sample Count", color="#5B8DEF")
ax2.set_ylabel("Hours", color="#F5A623")
axes[1].set_title("Distribution by Duration Bucket (Train)", fontweight="bold")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=10)

plt.tight_layout()
plt.savefig("viz/05_cumulative_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved viz/05_cumulative_coverage.png")

# ════════════════════════════════════════════════════════════════
#  Print Summary Report
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MYANMAR ASR DATASET — FULL SUMMARY REPORT")
print("="*60)

total_h = df["duration"].sum() / 3600
print(f"\n📊 TOTAL: {total_h:.2f} hours  ({total_h*60:.0f} minutes)")
print(f"   Total samples: {len(df):,}")
print(f"   Mean duration: {df['duration'].mean():.1f}s")
print(f"   Median duration: {df['duration'].median():.1f}s")
print(f"   Min: {df['duration'].min():.1f}s | Max: {df['duration'].max():.1f}s")

print("\n📁 BY SPLIT:")
for split_name in ["train", "validation", "test"]:
    if split_name in df["split"].values:
        dfs = df[df["split"] == split_name]
        h = dfs["duration"].sum() / 3600
        print(f"   {split_name:12s}: {len(dfs):>6,} samples | {h:>6.2f}h ({h*60:>5.0f} min)")

print("\n🗂️  BY SOURCE:")
for src in sorted(df["source"].unique()):
    dfs = df[df["source"] == src]
    h = dfs["duration"].sum() / 3600
    print(f"   {src:12s}: {len(dfs):>6,} samples | {h:>6.2f}h ({h*60:>5.0f} min) | "
          f"speakers: {dfs['speaker_id'].nunique()}")

print("\n🗂️  BY SOURCE × SPLIT:")
cross = df.groupby(["source", "split"]).agg(
    samples=("duration", "count"),
    hours=("duration", lambda x: x.sum()/3600)
).round(2)
print(cross.to_string())
print("\n" + "="*60)
