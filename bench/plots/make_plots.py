"""Render the ternfpga result figures from measured data into bench/plots/*.png.

Self-contained + reproducible: the numbers below are transcribed from the measured
results in bench/results/ (cited per-figure). Run:  python bench/plots/make_plots.py

Figures:
  1. energy_per_token.png    — CPU vs GPU (measured) vs FPGA target, BitNet-2B-4T batch-1
  2. activation_sparsity.png — BitNet b1.58 FFN sparsity by layer (measured) + relu-fication band
  3. fit_sweep.png           — LUT/FF/DSP vs FFN width on the 35T (0 DSP; FFs are the wall)
  4. bandwidth_roofline.png  — batch-1 tok/s ceiling vs model size at the Arty's DDR3 bandwidth
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

NAVY, STEEL, ORANGE, GREEN, GREY = "#1b2a4a", "#3b6ea5", "#e8743b", "#2e8b57", "#9aa3ad"
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def energy_per_token():
    """Source: bench/results/gpu_baseline.md (CPU 4.62, GPU 3.67 J/tok); FPGA target 0.25-0.40."""
    labels = ["CPU 5950X\n(native ternary)", "RTX 3060\n(bf16, dequantized)",
              "FPGA A7-35T\n(0-DSP ternary, target)"]
    vals = [4.62, 3.67, 0.33]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bars = ax.bar(labels, vals, color=[GREY, STEEL, ORANGE], width=0.6)
    ax.errorbar(2, 0.33, yerr=[[0.08], [0.07]], fmt="none", ecolor=NAVY, capsize=6, lw=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.09, f"{v:.2f}", ha="center",
                va="bottom", fontweight="bold")
    ax.text(2, 0.55, "target\n0.25–0.40", ha="center", va="bottom", fontsize=8, color=NAVY)
    ax.set_ylabel("Joules / token  (lower is better)")
    ax.set_title("Energy per token — BitNet-2B-4T, batch-1 decode")
    ax.set_ylim(0, 5.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("GPU has no ternary datapath → dequantizes\nto bf16, barely beats the CPU (and is slower)",
                xy=(1, 3.74), xytext=(1.5, 4.7), fontsize=8, color=NAVY, ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=NAVY))
    _save(fig, "energy_per_token.png")


def activation_sparsity():
    """Source: bench/results/activation_sparsity.md (per-layer measured, mean 59.8%)."""
    spars = [44.2, 73.4, 78.9, 70.3, 66.1, 66.6, 69.9, 67.4, 61.4, 55.5, 51.4, 46.5,
             45.7, 46.1, 43.7, 41.6, 47.2, 52.7, 55.2, 58.0, 62.8, 67.6, 63.9, 67.4,
             67.8, 66.5, 66.7, 65.4, 64.3, 59.9]
    layers = list(range(len(spars)))
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(layers, spars, color=STEEL, width=0.82)
    ax.axhline(59.8, color=ORANGE, ls="--", lw=1.6, label="measured mean 59.8%")
    ax.axhspan(85, 95, color=GREEN, alpha=0.15, label="relu-fication target 85–95%")
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("FFN activation sparsity (% zero / token)")
    ax.set_title("BitNet b1.58 FFN activation sparsity by layer (measured)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "activation_sparsity.png")


def fit_sweep():
    """Source: bench/results/fit_sweep.md (% of 35T budget; DSP=0 at every point)."""
    labels = ["32", "128", "1024\n(K=16)", "1024\n(K=32)", "2048"]
    lut = [2.7, 7.6, 49.2, 42.8, 52.9]
    ff = [2.1, 7.6, 59.3, 59.4, 79.2]
    dsp = [0, 0, 0, 0, 0]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(x - w, lut, w, label="LUT %", color=STEEL)
    ax.bar(x, ff, w, label="FF %", color=ORANGE)
    ax.bar(x + w, dsp, w, label="DSP %", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("FFN row width KT (lanes K)")
    ax.set_ylabel("% of Arty A7-35T budget")
    ax.set_title("Ternary GEMV fit on the 35T — 0 DSP everywhere; FFs are the wall")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("register-resident operands\n→ move to BRAM", xy=(4, 79.2), xytext=(2.4, 88),
                fontsize=8, color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="->", color=NAVY))
    _save(fig, "fit_sweep.png")


def bandwidth_roofline():
    """Batch-1 tok/s ceiling = BW / (bytes_per_token); 1.6 bit/weight => 0.2 B/param; BW~0.7 GB/s.
    Source: scaling-feasibility.md §2 (bandwidth wall)."""
    bw, bpp = 0.7e9, 0.2
    p = np.logspace(7, 9.9, 200)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.loglog(p / 1e6, bw / (bpp * p), color=NAVY, lw=2)
    for name, pv, dy in [("single block ~50M", 50e6, 8), ("0.7B", 0.7e9, 8),
                         ("2B", 2e9, 8), ("7B", 7e9, -14)]:
        t = bw / (bpp * pv)
        ax.scatter(pv / 1e6, t, color=ORANGE, zorder=5)
        ax.annotate(name, (pv / 1e6, t), textcoords="offset points", xytext=(6, dy), fontsize=8)
    ax.axhline(1, color=GREY, ls=":", lw=1)
    ax.text(11, 1.25, "1 tok/s", fontsize=8, color=GREY)
    ax.set_xlabel("model size (M params)")
    ax.set_ylabel("batch-1 decode ceiling (tok/s)")
    ax.set_title("DDR3 bandwidth roofline — Arty A7-35T (~0.7 GB/s, 1.6 bit/weight)")
    ax.grid(True, which="both", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "bandwidth_roofline.png")


if __name__ == "__main__":
    energy_per_token()
    activation_sparsity()
    fit_sweep()
    bandwidth_roofline()
    print("done — 4 figures in", OUT)
