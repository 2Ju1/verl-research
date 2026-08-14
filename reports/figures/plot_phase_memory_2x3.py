import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/mnt/sda/juwon/verl-research")
SUMMARY = ROOT / "outputs/pa-repro-fp32-v1/summary"

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update(
    {
        "font.family": font_name,
        "axes.unicode_minus": False,
        "font.size": 9,
    }
)

configs = ["GGG", "GGC", "GCG", "GCC", "CGG", "CGC", "CCG", "CCC"]
config_ids = {c: f"FP32-R-{c}" for c in configs}
config_labels = [
    "All on\nGPU",
    "Offload\nAdam",
    "Offload\nRef.",
    "Offload\nRef.+Adam",
    "Offload\nActor",
    "Offload\nActor+Adam",
    "Offload\nActor+Ref.",
    "Offload\nall",
]

phase_columns = [
    ("Rollout", "phase", "rollout"),
    ("Actor log-prob", "phase", "actor_log_prob"),
    ("Reference log-prob", "phase", "reference_log_prob"),
    ("Actor forward", "subphase", "actor_forward_end"),
    ("Actor backward", "subphase", "actor_backward_end"),
    ("Optimize", "subphase", "actor_optimizer_end"),
]

phase_values = {}
with open(SUMMARY / "phase_configs.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["config_id"] in config_ids.values():
            phase_values[(row["config_id"], row["phase"])] = float(
                row["allocator_peak_allocated_gb_mean"]
            )

subphase_values = {}
with open(SUMMARY / "actor_subphase_configs.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["config_id"] in config_ids.values():
            subphase_values[(row["config_id"], row["tag"])] = float(
                row["gpu_peak_allocated_gb_mean"]
            )

values = np.zeros((len(configs), len(phase_columns)))
for i, config in enumerate(configs):
    config_id = config_ids[config]
    for j, (_, source, key) in enumerate(phase_columns):
        table = phase_values if source == "phase" else subphase_values
        values[i, j] = table[(config_id, key)]

colors = ["#315F8C", "#6F97B8", "#A8C4D8", "#F0B27A", "#D76A4A", "#8F3F36"]
hatches = ["", "", "", "", "", ""]

fig, ax = plt.subplots(figsize=(12.0, 5.3), facecolor="white")
x = np.arange(len(configs))
group_width = 0.82
bar_width = group_width / len(phase_columns)

for j, ((label, _, _), color, hatch) in enumerate(zip(phase_columns, colors, hatches)):
    offset = (j - (len(phase_columns) - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        values[:, j],
        width=bar_width * 0.92,
        label=label,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        hatch=hatch,
    )

ax.set_xlim(-0.55, len(configs) - 0.45)
ax.set_ylim(0, 11.1)
ax.set_ylabel("Peak allocated memory (GiB)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(config_labels, fontsize=9)
ax.set_yticks(np.arange(0, 12, 2))
ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.65)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#9AA6B2")
ax.spines["bottom"].set_color("#9AA6B2")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=6,
    frameon=False,
    fontsize=8.5,
    handlelength=1.5,
    columnspacing=1.25,
)

fig.tight_layout(pad=0.8)
fig.subplots_adjust(bottom=0.24)

out = ROOT / "reports/figures/phase_memory_2x3_05b"
fig.savefig(str(out) + ".png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(str(out) + ".svg", bbox_inches="tight", facecolor="white")
print(str(out) + ".png")
print(str(out) + ".svg")
