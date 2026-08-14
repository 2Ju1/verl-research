import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/data/02_allgpu_vs_phase_offload"

font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
else:
    font_name = "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": font_name,
        "axes.unicode_minus": False,
        "font.size": 10,
        "svg.hashsalt": "verl-result-figures",
    }
)

configs = {
    "All on GPU": "FP32-LATE-GGG",
    "Phase offload": "FP32-LATE-CCC",
}
phases = [
    ("Rollout", "phase", "rollout"),
    ("Actor\nlog-prob", "phase", "actor_log_prob"),
    ("Reference\nlog-prob", "phase", "reference_log_prob"),
    ("Actor\nforward", "subphase", "actor_forward_end"),
    ("Actor\nbackward", "subphase", "actor_backward_end"),
    ("Update", "subphase", "actor_optimizer_end"),
]

phase_values = {}
with open(SUMMARY / "phase_configs.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["config_id"] in configs.values():
            phase_values[(row["config_id"], row["phase"])] = float(
                row["allocator_peak_allocated_gb_mean"]
            )

subphase_values = {}
with open(SUMMARY / "actor_subphase_configs.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["config_id"] in configs.values():
            subphase_values[(row["config_id"], row["tag"])] = float(
                row["gpu_peak_allocated_gb_mean"]
            )

data = {}
for label, config_id in configs.items():
    data[label] = []
    for _, source, key in phases:
        table = phase_values if source == "phase" else subphase_values
        data[label].append(table[(config_id, key)])

x = np.arange(len(phases))
width = 0.34
fig, ax = plt.subplots(figsize=(8.6, 4.7), facecolor="white")

bars_gpu = ax.bar(
    x - width / 2,
    data["All on GPU"],
    width,
    label="All on GPU",
    color="#A9B9CA",
    edgecolor="white",
    linewidth=0.6,
)
bars_offload = ax.bar(
    x + width / 2,
    data["Phase offload"],
    width,
    label="Phase offload",
    color="#315F8C",
    edgecolor="white",
    linewidth=0.6,
)

for bars in (bars_gpu, bars_offload):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.11,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#374151",
        )

ax.set_xlim(-0.55, len(phases) - 0.45)
ax.set_ylim(0, 11.1)
ax.set_ylabel("Peak allocated memory (GiB)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in phases], fontsize=9.5)
ax.set_yticks(np.arange(0, 12, 2))
ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.65)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#9AA6B2")
ax.spines["bottom"].set_color("#9AA6B2")
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9.5)

fig.tight_layout(pad=0.9)
out = ROOT / "reports/figures/allgpu_vs_phase_offload_05b"
fig.savefig(
    str(out) + ".png",
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
    metadata={"Software": "verl-research"},
)
fig.savefig(
    str(out) + ".svg",
    bbox_inches="tight",
    facecolor="white",
    metadata={"Date": None},
)
print(str(out) + ".png")
print(str(out) + ".svg")
