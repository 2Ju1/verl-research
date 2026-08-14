#!/usr/bin/env python3
"""Regenerate the three compact result figures used by the study report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
LIGHT = "#a8bacd"
DARK = "#376b98"
GRID = "#d9e2ec"
plt.rcParams["svg.hashsalt"] = "verl-result-figures"


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(
        OUT / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "verl-research"},
    )
    fig.savefig(
        OUT / f"{name}.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def label_bars(ax, bars, digits=2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def cpu_adam():
    # phase-best-vs-cpu-adamw-memory/performance-v1
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    names = ["Phase offload\n(GPU AdamW)", "Phase offload\n+ CPU AdamW"]
    colors = [LIGHT, DARK]
    memory = axes[0].bar(names, [8.408, 2.365], color=colors, width=0.62)
    timing = axes[1].bar(names, [0.129, 3.565], color=colors, width=0.62)
    axes[0].set_title("Peak Allocated GPU Memory")
    axes[0].set_ylabel("Update peak (GiB)")
    axes[1].set_title("Phase Execution Time")
    axes[1].set_ylabel("Update time (s)")
    label_bars(axes[0], memory)
    label_bars(axes[1], timing)
    for ax in axes:
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "result_cpu_adamw")


def streaming():
    # no-stream direct remeasure + optimized 16 MiB overlap performance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["No-stream", "16 MiB streaming"]
    colors = [LIGHT, DARK]
    memory = axes[0].bar(names, [4.721, 3.402], color=colors, width=0.62)
    axes[0].set_title("Peak Allocated GPU Memory")
    axes[0].set_ylabel("Actor backward peak (GiB)")
    label_bars(axes[0], memory)

    x = np.arange(2)
    width = 0.34
    backward = axes[1].bar(x - width / 2, [0.212, 0.292], width, label="Actor backward", color=LIGHT)
    update = axes[1].bar(x + width / 2, [3.551, 3.020], width, label="Update", color=DARK)
    axes[1].set_xticks(x, names)
    axes[1].set_title("Phase Execution Time")
    axes[1].set_ylabel("Time per training step (s)")
    axes[1].legend(frameon=False)
    label_bars(axes[1], backward)
    label_bars(axes[1], update)
    for ax in axes:
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "result_gradient_streaming")


def capacity():
    # OOM snapshot-v2 + pa-capacity-fp32-qwen15b-v1/detail
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = ["All GPU", "CPU AdamW", "CPU AdamW\n+ stream"]
    values = [11.34, 11.34, 8.44]
    bars = ax.bar(names, values, color=[LIGHT, LIGHT, DARK], width=0.62)
    bars[0].set_hatch("///")
    bars[1].set_hatch("///")
    ax.axhline(11.90, color="#64748b", linestyle="--", linewidth=1.5, label="GPU capacity 11.90 GiB")
    for index, bar in enumerate(bars):
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.8,
            "OOM" if index < 2 else "SUCCESS",
            ha="center",
            va="center",
            color="#c24141" if index < 2 else "white",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_title("Qwen2.5-1.5B FP32 Training on a 12 GiB GPU")
    ax.set_ylabel("Peak allocated GPU memory (GiB)")
    ax.set_ylim(0, 12.7)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "result_qwen15b_capacity")


if __name__ == "__main__":
    cpu_adam()
    streaming()
    capacity()
