import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager


font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update(
    {
        "font.family": font_name,
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)

rows = [
    # placement, step time, throughput, allocated peak, device peak
    ("GGG", 4.505, 147.9, 10.242, 11.355),
    ("GGC", 5.328, 125.1, 10.246, 11.331),
    ("GCG", 4.956, 134.5, 8.404, 9.485),
    ("GCC", 5.733, 116.3, 8.405, 9.392),
    ("CGG", 5.527, 120.6, 10.248, 11.114),
    ("CGC", 6.439, 103.3, 10.246, 11.347),
    ("CCG", 5.922, 112.6, 8.406, 9.175),
    ("CCC", 6.812, 98.1, 8.403, 9.280),
]

colors = {"G": "#1665A8", "C": "#38A169"}
labels = {
    "GGG": "All on\nGPU",
    "GGC": "Offload\nAdam",
    "GCG": "Offload\nRef.",
    "GCC": "Offload\nRef.+Adam",
    "CGG": "Offload\nActor",
    "CGC": "Offload\nActor+Adam",
    "CCG": "Offload\nActor+Ref.",
    "CCC": "Offload\nall",
}
fig, ax = plt.subplots(figsize=(9.0, 4.5), facecolor="white")

for placement, step, throughput, allocated, device in rows:
    ref = placement[1]
    ax.scatter(
        step,
        allocated,
        s=72,
        c=colors[ref],
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax.annotate(
        labels[placement],
        (step, allocated),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="normal",
        color="#374151",
    )

ax.set_xlim(4.25, 7.05)
ax.set_ylim(8.15, 10.55)
ax.set_xlabel("Step time (s)", fontsize=11)
ax.set_ylabel("Peak allocated memory (GiB)", fontsize=11)
ax.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
ax.set_yticks([8.4, 8.8, 9.2, 9.6, 10.0, 10.4])
ax.grid(axis="both", color="#D7DEE7", linewidth=0.65, alpha=0.55)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#9AA6B2")
ax.spines["bottom"].set_color("#9AA6B2")

legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["G"], markeredgecolor="white", markersize=8, label="Reference on GPU"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["C"], markeredgecolor="white", markersize=8, label="Reference offloaded"),
]
ax.legend(handles=legend, loc="center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=9)
fig.tight_layout(pad=0.8)

out = "/mnt/sda/juwon/verl-research/reports/figures/placement_pareto_05b"
fig.savefig(out + ".png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out + ".svg", bbox_inches="tight", facecolor="white")
print(out + ".png")
print(out + ".svg")
