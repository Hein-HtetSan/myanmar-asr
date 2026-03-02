#!/usr/bin/env python3
"""
Myanmar ASR — Presentation Visualization Suite
Generates publication-ready charts for professor presentation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json, os

os.makedirs("viz", exist_ok=True)

# ── Load MLflow data ──────────────────────────────────────────
with open("viz/mlflow_metrics.json") as f:
    data = json.load(f)

# Short names for plots
SHORT = {
    "Whisper Turbo v3": "Whisper Turbo v3",
    "Dolphin (Whisper-large-v2)": "Dolphin\n(Whisper-large-v2)",
    "SeamlessM4T v2 Large": "SeamlessM4T v2",
}
COLORS = {
    "Whisper Turbo v3": "#E74C3C",
    "Dolphin (Whisper-large-v2)": "#5B8DEF",
    "SeamlessM4T v2 Large": "#2ECC71",
}

# Best metrics per model
BEST = {}
for name, d in data.items():
    wers = [v for _, v in d["eval_wer"]] if d["eval_wer"] else [999]
    cers = [v for _, v in d["eval_cer"]] if d["eval_cer"] else [999]
    BEST[name] = {
        "wer": min(wers),
        "cer": min(cers),
        "status": d["status"],
    }

print("Model summary:")
for n, b in BEST.items():
    tag = " (training)" if b["status"] == "RUNNING" else ""
    print(f"  {n}: WER={b['wer']:.1f}%, CER={b['cer']:.1f}%{tag}")


# ════════════════════════════════════════════════════════════════
#  CHART 6: Model Comparison Bar Chart (WER & CER side by side)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")

models = list(data.keys())
x = np.arange(len(models))
width = 0.32

wers = [BEST[m]["wer"] for m in models]
cers = [BEST[m]["cer"] for m in models]
colors_wer = [COLORS[m] for m in models]
colors_cer = [plt.cm.Set2(i) for i in range(len(models))]

bars1 = ax.bar(x - width/2, wers, width, label="WER (%)",
               color=[COLORS[m] for m in models], edgecolor="white", linewidth=1.5, alpha=0.9)
bars2 = ax.bar(x + width/2, cers, width, label="CER (%)",
               color=[COLORS[m] for m in models], edgecolor="white", linewidth=1.5, alpha=0.5,
               hatch="///")

for bar, val in zip(bars1, wers):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%",
            ha="center", fontweight="bold", fontsize=13, color="#333")
for bar, val in zip(bars2, cers):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%",
            ha="center", fontweight="bold", fontsize=13, color="#666")

labels = []
for m in models:
    lbl = SHORT[m]
    if BEST[m]["status"] == "RUNNING":
        lbl += "\n(in progress)"
    labels.append(lbl)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Error Rate (%)", fontsize=13)
ax.set_title("Myanmar ASR — Model Comparison (Best Validation Scores)", fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=12, loc="upper right")
ax.set_ylim(0, max(wers) * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Highlight best
best_idx = wers.index(min(wers))
ax.annotate("BEST WER", xy=(best_idx - width/2, min(wers)),
            xytext=(best_idx - width/2, min(wers) + 12),
            ha="center", fontsize=11, fontweight="bold", color="#27AE60",
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=2))

plt.tight_layout()
plt.savefig("viz/06_model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/06_model_comparison_bar.png")


# ════════════════════════════════════════════════════════════════
#  CHART 7: WER Learning Curves (all models on one plot)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("white")
fig.suptitle("Myanmar ASR — Training Progress (Eval Metrics Over Steps)",
             fontsize=16, fontweight="bold", y=1.02)

# 7a. WER curves
for name, d in data.items():
    if not d["eval_wer"]:
        continue
    steps = [s for s, _ in d["eval_wer"]]
    vals = [v for _, v in d["eval_wer"]]
    label = SHORT[name].replace("\n", " ")
    if d["status"] == "RUNNING":
        label += " (training)"
    axes[0].plot(steps, vals, marker="o", markersize=4, linewidth=2.2,
                 color=COLORS[name], label=label, alpha=0.9)
    # Mark best point
    best_i = vals.index(min(vals))
    axes[0].scatter([steps[best_i]], [vals[best_i]], s=120, color=COLORS[name],
                    zorder=5, edgecolors="white", linewidths=2)
    axes[0].annotate(f"{vals[best_i]:.1f}%", (steps[best_i], vals[best_i]),
                     textcoords="offset points", xytext=(10, -10),
                     fontsize=10, fontweight="bold", color=COLORS[name])

axes[0].set_xlabel("Training Step", fontsize=12)
axes[0].set_ylabel("Word Error Rate (%)", fontsize=12)
axes[0].set_title("WER Over Training Steps", fontweight="bold", fontsize=13)
axes[0].legend(fontsize=10, loc="upper right")
axes[0].grid(alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# 7b. CER curves
for name, d in data.items():
    if not d["eval_cer"]:
        continue
    steps = [s for s, _ in d["eval_cer"]]
    vals = [v for _, v in d["eval_cer"]]
    label = SHORT[name].replace("\n", " ")
    if d["status"] == "RUNNING":
        label += " (training)"
    axes[1].plot(steps, vals, marker="s", markersize=4, linewidth=2.2,
                 color=COLORS[name], label=label, alpha=0.9)
    best_i = vals.index(min(vals))
    axes[1].scatter([steps[best_i]], [vals[best_i]], s=120, color=COLORS[name],
                    zorder=5, edgecolors="white", linewidths=2)
    axes[1].annotate(f"{vals[best_i]:.1f}%", (steps[best_i], vals[best_i]),
                     textcoords="offset points", xytext=(10, -10),
                     fontsize=10, fontweight="bold", color=COLORS[name])

axes[1].set_xlabel("Training Step", fontsize=12)
axes[1].set_ylabel("Character Error Rate (%)", fontsize=12)
axes[1].set_title("CER Over Training Steps", fontweight="bold", fontsize=13)
axes[1].legend(fontsize=10, loc="upper right")
axes[1].grid(alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("viz/07_training_curves_wer_cer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/07_training_curves_wer_cer.png")


# ════════════════════════════════════════════════════════════════
#  CHART 8: Train Loss Curves (all models)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

for name, d in data.items():
    if not d["train_loss"]:
        continue
    steps = [s for s, _ in d["train_loss"]]
    vals = [v for _, v in d["train_loss"]]
    label = SHORT[name].replace("\n", " ")
    if d["status"] == "RUNNING":
        label += " (training)"
    ax.plot(steps, vals, linewidth=1.8, color=COLORS[name], label=label, alpha=0.8)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.set_title("Myanmar ASR — Training Loss Convergence", fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("viz/08_train_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/08_train_loss_curves.png")


# ════════════════════════════════════════════════════════════════
#  CHART 9: Model Summary Table (as image)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("white")
ax.axis("off")
ax.set_title("Myanmar ASR — Model Comparison Summary",
             fontsize=16, fontweight="bold", pad=20)

col_labels = ["Model", "Base Model", "Params\n(Trainable)", "Best WER\n(%)",
              "Best CER\n(%)", "Train Time\n(min)", "Status"]

table_data = [
    ["Whisper Turbo v3", "openai/whisper-\nlarge-v3-turbo", "809M\n(809M)",
     f"{BEST['Whisper Turbo v3']['wer']:.1f}", f"{BEST['Whisper Turbo v3']['cer']:.1f}",
     "159", "Completed"],
    ["Dolphin", "openai/whisper-\nlarge-v2", "1.5B\n(636M)",
     f"{BEST['Dolphin (Whisper-large-v2)']['wer']:.1f}",
     f"{BEST['Dolphin (Whisper-large-v2)']['cer']:.1f}",
     "349", "Completed"],
    ["SeamlessM4T v2", "facebook/seamless-\nm4t-v2-large", "1.5B\n(867M)",
     f"{BEST['SeamlessM4T v2 Large']['wer']:.1f}",
     f"{BEST['SeamlessM4T v2 Large']['cer']:.1f}",
     "242", "Completed"],
]

table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc="center", loc="center",
                 colWidths=[0.16, 0.16, 0.12, 0.1, 0.1, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Style header row
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)

# Color rows
row_colors = ["#FDEBD0", "#D6EAF8", "#D5F5E3"]
for i in range(len(table_data)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(row_colors[i])
        cell.set_edgecolor("#BDC3C7")

# Highlight best WER cell
best_wer_row = min(range(len(table_data)), key=lambda i: float(table_data[i][3]))
table[best_wer_row + 1, 3].set_text_props(fontweight="bold", color="#27AE60")
table[best_wer_row + 1, 3].set_facecolor("#A9DFBF")

# Highlight best CER cell
best_cer_row = min(range(len(table_data)), key=lambda i: float(table_data[i][4]))
table[best_cer_row + 1, 4].set_text_props(fontweight="bold", color="#27AE60")
table[best_cer_row + 1, 4].set_facecolor("#A9DFBF")

ax.text(0.5, -0.05, "All models completed. Dolphin achieves best WER (33.0%); SeamlessM4T achieves best CER (12.6%). Test set: 319 samples.",
        ha="center", fontsize=10, style="italic", color="#666", transform=ax.transAxes)

plt.tight_layout()
plt.savefig("viz/09_model_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/09_model_summary_table.png")


# ════════════════════════════════════════════════════════════════
#  CHART 10: Radar / Spider Chart — Model Strengths
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

categories = ["WER\n(lower=better)", "CER\n(lower=better)", "Training\nSpeed",
              "Model Size\n(smaller=better)", "Convergence\nSpeed"]
N = len(categories)

# Normalize scores 0-100 (higher = better)
def invert_score(val, worst, best):
    """Convert metric where lower=better to 0-100 scale where higher=better."""
    return max(0, min(100, 100 * (worst - val) / (worst - best)))

# Raw values
raw = {
    "Whisper Turbo v3": {
        "wer": BEST["Whisper Turbo v3"]["wer"],
        "cer": BEST["Whisper Turbo v3"]["cer"],
        "train_min": 159,
        "params_b": 0.809,
        "converge_epochs": 10,
    },
    "Dolphin (Whisper-large-v2)": {
        "wer": BEST["Dolphin (Whisper-large-v2)"]["wer"],
        "cer": BEST["Dolphin (Whisper-large-v2)"]["cer"],
        "train_min": 349,
        "params_b": 1.5,
        "converge_epochs": 8,
    },
    "SeamlessM4T v2 Large": {
        "wer": BEST["SeamlessM4T v2 Large"]["wer"],
        "cer": BEST["SeamlessM4T v2 Large"]["cer"],
        "train_min": 242,
        "params_b": 1.5,
        "converge_epochs": 9,
    },
}

scores = {}
for name, r in raw.items():
    scores[name] = [
        invert_score(r["wer"], 100, 30),
        invert_score(r["cer"], 90, 10),
        invert_score(r["train_min"], 400, 100),
        invert_score(r["params_b"], 2.0, 0.5),
        invert_score(r["converge_epochs"], 15, 3),
    ]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for name in raw:
    vals = scores[name] + scores[name][:1]
    label = SHORT[name].replace("\n", " ")
    ax.plot(angles, vals, "o-", linewidth=2.2, color=COLORS[name], label=label, markersize=6)
    ax.fill(angles, vals, alpha=0.1, color=COLORS[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="#888")
ax.set_title("Model Strengths Comparison\n(higher = better)", fontsize=14,
             fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=10)

plt.tight_layout()
plt.savefig("viz/10_radar_model_strengths.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/10_radar_model_strengths.png")


# ════════════════════════════════════════════════════════════════
#  CHART 11: Pipeline Architecture Diagram
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("white")
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.set_title("Myanmar ASR — End-to-End Pipeline Architecture",
             fontsize=16, fontweight="bold", pad=15)

boxes = [
    (5, 12, "Data\nCollection", "#3498DB", "FLEURS\nOpenSLR-80\nYODAS"),
    (22, 12, "Data\nCleaning", "#9B59B6", "SNR filter\nDuration filter\nText normalize"),
    (39, 12, "Augmentation", "#E67E22", "Speed 0.9x\nSpeed 1.1x\n22.7K samples"),
    (56, 12, "Fine-tuning", "#E74C3C", "3 models\nRTX 4090\nMLflow tracking"),
    (73, 12, "Evaluation", "#27AE60", "WER / CER\nTest set\n319 samples"),
    (90, 12, "Deployment", "#95A5A6", "HF Hub\nAPI ready\n(next phase)"),
]

for x, y, title, color, desc in boxes:
    rect = plt.Rectangle((x - 7, y - 5), 14, 14, linewidth=2,
                          edgecolor=color, facecolor=color, alpha=0.15,
                          transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    rect2 = plt.Rectangle((x - 7, y + 5), 14, 4, linewidth=0,
                           facecolor=color, alpha=0.85,
                           transform=ax.transData, zorder=3)
    ax.add_patch(rect2)
    ax.text(x, y + 7, title, ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", zorder=4)
    ax.text(x, y - 1, desc, ha="center", va="center", fontsize=8,
            color="#333", zorder=4)

# Arrows between boxes
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 7
    x2 = boxes[i + 1][0] - 7
    ax.annotate("", xy=(x2, 12), xytext=(x1, 12),
                arrowprops=dict(arrowstyle="->", color="#666", lw=2.5),
                zorder=1)

# Bottom stats bar
stats_text = ("54.2 hours total  |  22,705 samples  |  3 sources  |  "
              "3 models fine-tuned  |  Best WER: 33.0%  |  Best CER: 12.6%")
ax.text(50, 1, stats_text, ha="center", va="center", fontsize=10,
        color="#555", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#DEE2E6"))

plt.tight_layout()
plt.savefig("viz/11_pipeline_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/11_pipeline_architecture.png")


# ════════════════════════════════════════════════════════════════
#  CHART 12: Data Processing Funnel
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("white")
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title("Myanmar ASR — Data Processing Pipeline",
             fontsize=16, fontweight="bold", pad=15)

funnel_steps = [
    ("Raw Collection", "17,552 samples", "~50h audio", "#3498DB", 90),
    ("Quality Filtering", "12,299 samples", "~24.5h (SNR, duration, text)", "#9B59B6", 70),
    ("Train/Val/Test Split", "11,661 / 319 / 319", "Stratified by source", "#E67E22", 60),
    ("Speed Augmentation", "20,814 train samples", "0.9x + 1.1x speed", "#E74C3C", 80),
    ("Final Dataset", "22,705 total", "54.2h (3 splits)", "#27AE60", 95),
]

y_positions = [82, 66, 50, 34, 18]
for i, (title, count, detail, color, width_pct) in enumerate(funnel_steps):
    y = y_positions[i]
    w = width_pct * 0.45
    rect = plt.Rectangle((50 - w, y - 4), 2 * w, 8, linewidth=0,
                          facecolor=color, alpha=0.75, transform=ax.transData,
                          zorder=2)
    ax.add_patch(rect)
    ax.text(50, y + 1, title, ha="center", va="center", fontsize=12,
            fontweight="bold", color="white", zorder=3)
    ax.text(50, y - 2, f"{count}  —  {detail}", ha="center", va="center",
            fontsize=9, color="white", zorder=3, alpha=0.9)

    if i < len(funnel_steps) - 1:
        ax.annotate("", xy=(50, y_positions[i + 1] + 5), xytext=(50, y - 5),
                     arrowprops=dict(arrowstyle="->", color="#888", lw=2))

plt.tight_layout()
plt.savefig("viz/12_data_processing_funnel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/12_data_processing_funnel.png")


# ════════════════════════════════════════════════════════════════
#  CHART 13: Eval Loss Curves
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

for name, d in data.items():
    if not d["eval_loss"]:
        continue
    steps = [s for s, _ in d["eval_loss"]]
    vals = [v for _, v in d["eval_loss"]]
    label = SHORT[name].replace("\n", " ")
    if d["status"] == "RUNNING":
        label += " (training)"
    ax.plot(steps, vals, marker="o", markersize=4, linewidth=2.2,
            color=COLORS[name], label=label, alpha=0.9)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Eval Loss", fontsize=12)
ax.set_title("Myanmar ASR — Validation Loss Over Training", fontsize=16,
             fontweight="bold", pad=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("viz/13_eval_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/13_eval_loss_curves.png")


# ════════════════════════════════════════════════════════════════
#  CHART 14: Final Test Results Bar Chart
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")

# Test results from MLflow (test_wer, test_cer from final eval)
test_results = {
    "Whisper Turbo v3": {"test_wer": 53.50, "test_cer": 34.78, "test_loss": 1.471},
    "Dolphin (Whisper-large-v2)": {"test_wer": 33.02, "test_cer": 28.00, "test_loss": 1.451},
    "SeamlessM4T v2 Large": {"test_wer": 49.12, "test_cer": 13.04, "test_loss": 2.070},
}

models_t = list(test_results.keys())
x = np.arange(len(models_t))
width = 0.32

test_wers = [test_results[m]["test_wer"] for m in models_t]
test_cers = [test_results[m]["test_cer"] for m in models_t]

bars1 = ax.bar(x - width/2, test_wers, width, label="Test WER (%)",
               color=[COLORS[m] for m in models_t], edgecolor="white", linewidth=1.5, alpha=0.9)
bars2 = ax.bar(x + width/2, test_cers, width, label="Test CER (%)",
               color=[COLORS[m] for m in models_t], edgecolor="white", linewidth=1.5, alpha=0.5,
               hatch="///")

for bar, val in zip(bars1, test_wers):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val:.1f}%",
            ha="center", fontweight="bold", fontsize=13, color="#333")
for bar, val in zip(bars2, test_cers):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val:.1f}%",
            ha="center", fontweight="bold", fontsize=13, color="#666")

labels_t = [SHORT[m].replace("\n", " ") for m in models_t]
ax.set_xticks(x)
ax.set_xticklabels(labels_t, fontsize=12)
ax.set_ylabel("Error Rate (%)", fontsize=13)
ax.set_title("Myanmar ASR — Final Test Set Results (All Models Completed)",
             fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=12, loc="upper right")
ax.set_ylim(0, max(test_wers) * 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Highlight best WER
best_wer_idx = test_wers.index(min(test_wers))
ax.annotate("BEST WER", xy=(best_wer_idx - width/2, min(test_wers)),
            xytext=(best_wer_idx - width/2, min(test_wers) + 10),
            ha="center", fontsize=11, fontweight="bold", color="#27AE60",
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=2))

# Highlight best CER
best_cer_idx = test_cers.index(min(test_cers))
ax.annotate("BEST CER", xy=(best_cer_idx + width/2, min(test_cers)),
            xytext=(best_cer_idx + width/2, min(test_cers) + 10),
            ha="center", fontsize=11, fontweight="bold", color="#8E44AD",
            arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=2))

plt.tight_layout()
plt.savefig("viz/14_test_results_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/14_test_results_bar.png")


# ════════════════════════════════════════════════════════════════
#  CHART 15: Final Summary Scorecard
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("#FAFBFC")
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Title
ax.text(50, 95, "Myanmar ASR — Final Results Summary", ha="center", va="center",
        fontsize=22, fontweight="bold", color="#2C3E50")
ax.text(50, 89, "Comparative Fine-Tuning of 3 Multilingual Speech Models on 54.2h Myanmar Data",
        ha="center", va="center", fontsize=12, color="#7F8C8D", style="italic")

# Model cards
cards = [
    {"name": "Whisper Turbo v3", "base": "openai/whisper-large-v3-turbo",
     "wer": 53.50, "cer": 34.78, "time": "159 min", "color": "#E74C3C", "x": 17},
    {"name": "Dolphin", "base": "openai/whisper-large-v2",
     "wer": 33.02, "cer": 28.00, "time": "349 min", "color": "#5B8DEF", "x": 50},
    {"name": "SeamlessM4T v2", "base": "facebook/seamless-m4t-v2-large",
     "wer": 49.12, "cer": 13.04, "time": "242 min", "color": "#2ECC71", "x": 83},
]

for c in cards:
    cx, cy = c["x"], 58
    # Card background
    rect = plt.Rectangle((cx - 14, cy - 22), 28, 44, linewidth=2,
                          edgecolor=c["color"], facecolor="white",
                          transform=ax.transData, zorder=2, alpha=0.95)
    ax.add_patch(rect)
    # Header bar
    rect2 = plt.Rectangle((cx - 14, cy + 16), 28, 6, linewidth=0,
                           facecolor=c["color"], alpha=0.9,
                           transform=ax.transData, zorder=3)
    ax.add_patch(rect2)
    # Model name
    ax.text(cx, cy + 19, c["name"], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=4)
    # Base model
    ax.text(cx, cy + 12, c["base"], ha="center", va="center",
            fontsize=8, color="#888", zorder=4)
    # WER
    ax.text(cx, cy + 4, f"Test WER", ha="center", va="center",
            fontsize=9, color="#888", zorder=4)
    ax.text(cx, cy - 2, f"{c['wer']:.1f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color=c["color"], zorder=4)
    # CER
    ax.text(cx, cy - 10, f"Test CER", ha="center", va="center",
            fontsize=9, color="#888", zorder=4)
    ax.text(cx, cy - 16, f"{c['cer']:.1f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#555", zorder=4)
    # Train time
    ax.text(cx, cy - 21, f"Train: {c['time']}", ha="center", va="center",
            fontsize=9, color="#AAA", zorder=4)

# Winner badges
ax.text(50, 31, "BEST WER", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#5B8DEF",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6EAF8", edgecolor="#5B8DEF", linewidth=2))
ax.text(83, 31, "BEST CER", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#2ECC71",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5F5E3", edgecolor="#2ECC71", linewidth=2))

# Key findings
findings = [
    "Dataset: 22,705 samples (54.2h) from FLEURS + OpenSLR-80 + YODAS + augmentation",
    "Hardware: NVIDIA RTX 4090 (24GB) on Vast.ai  |  Framework: HuggingFace Transformers 5.2.0",
    "Key finding: Dolphin (Whisper-large-v2) achieves best word accuracy; SeamlessM4T achieves best character accuracy",
]
for i, text in enumerate(findings):
    ax.text(50, 19 - i * 5, text, ha="center", va="center",
            fontsize=9.5, color="#555", zorder=4)

ax.text(50, 3, "All training completed  |  March 2026  |  Experiment tracking: MLflow  |  Hein Htet San",
        ha="center", va="center", fontsize=9, color="#AAA")

plt.tight_layout()
plt.savefig("viz/15_final_summary_scorecard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved viz/15_final_summary_scorecard.png")


# ════════════════════════════════════════════════════════════════
#  Done!
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("  ALL PRESENTATION CHARTS GENERATED!")
print("=" * 50)
print("\nFiles in viz/:")
print("  01_overview_dashboard.png      (dataset analysis)")
print("  02_hours_splits_summary.png    (dataset analysis)")
print("  03_per_source_analysis.png     (dataset analysis)")
print("  04_duration_vs_text.png        (dataset analysis)")
print("  05_cumulative_coverage.png     (dataset analysis)")
print("  06_model_comparison_bar.png    (WER/CER bars)")
print("  07_training_curves_wer_cer.png (WER/CER over steps)")
print("  08_train_loss_curves.png       (loss convergence)")
print("  09_model_summary_table.png     (comparison table)")
print("  10_radar_model_strengths.png   (spider chart)")
print("  11_pipeline_architecture.png   (pipeline diagram)")
print("  12_data_processing_funnel.png  (data funnel)")
print("  13_eval_loss_curves.png        (eval loss curves)")
