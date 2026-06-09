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


def bram_fix():
    """Source: fit_sweep.md (register-resident tiled @ width 2048) + gemv_stream.md (BRAM stream)."""
    metrics = ["LUT %", "FF %"]
    tiled = [52.9, 79.2]
    stream = [2.3, 0.9]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    b1 = ax.bar(x - w / 2, tiled, w, label="register-resident (flat NT:1 mux)", color=GREY)
    b2 = ax.bar(x + w / 2, stream, w, label="BRAM-centric (sequential stream)", color=GREEN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("% of Arty A7-35T budget")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("The fit-sweep fix: register-resident → BRAM-centric ternary GEMV")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_ylim(0, 95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 44, "timing @ 100 MHz, both 0 DSP:\nflat mux   −5.9 ns  FAIL  (~63 MHz)\n"
            "BRAM       +3.5 ns  PASS  (~154 MHz)", fontsize=8, color=NAVY, ha="center",
            bbox=dict(boxstyle="round", fc="white", ec=GREY))
    _save(fig, "bram_fix.png")


def gather_savings():
    """Source: sim/tb_gemv_gather (down_proj column-sparse gather) + activation_sparsity.md."""
    dens = np.linspace(0.05, 1.0, 100)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(dens * 100, dens * 100, color=NAVY, lw=2, label="gathered (column-sparse)")
    ax.plot([5, 100], [100, 100], color=GREY, ls="--", lw=1.5,
            label="dense / GPU (no per-token skip)")
    sx = [100, 60, 40.2, 15]                    # density %
    sy = [100, 62.5, 43.8, 18.8]                # measured fetched % (sim, incl. K=16 padding)
    ax.scatter(sx, sy, color=ORANGE, zorder=5)
    ax.annotate("BitNet b1.58 measured\n~40% active → 56% saved", (40.2, 43.8),
                textcoords="offset points", xytext=(10, 26), fontsize=8, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.annotate("relu-fied ~15%\n→ 81% saved", (15, 18.8), textcoords="offset points",
                xytext=(26, 36), fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("FFN activation density (% nonzero per token)")
    ax.set_ylabel("down_proj weight bytes fetched (% of dense)")
    ax.set_title("Activation-sparse gather: fetch scales with density (down_proj)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 108)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "gather_savings.png")


def sparsity_compare():
    """Source: activation_sparsity.md (BitNet 59.8%) + relu_fication_upside.md (ProSparse 83.3% measured)."""
    models = ["BitNet b1.58 2B-4T\n(squared-ReLU)", "ProSparse-Llama-2-7B\n(ReLU, relu-fied)"]
    spars = [59.8, 83.3]
    saved = [56, 80]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhspan(85, 95, color=GREEN, alpha=0.13, label="relu-fication literature (85–95%)")
    b1 = ax.bar(x - w / 2, spars, w, label="FFN activation sparsity (% zero)", color=STEEL)
    b2 = ax.bar(x + w / 2, saved, w, label="down_proj bytes saved (gather)", color=ORANGE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("%")
    ax.set_title("Direction-D upside: relu-fication ~doubles the gather payoff (measured)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "sparsity_compare.png")


def ffn_block_energy():
    """Source: onboard_throughput_measured.md (measured 1 tile/cycle, 0.489 W SoC) + gpu_baseline.md."""
    labels = ["RTX 3060\n(extrapolated)", "FPGA SoC\n(measured)", "FPGA SoC\n+ gather",
              "FPGA engine\n(0-DSP, est.)"]
    vals = [61, 32, 26, 5]
    colors = [STEEL, ORANGE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("energy per FFN block (mJ, lower is better)")
    ax.set_title("Energy / BitNet-2B FFN block — measured 1 tile/cycle × power")
    ax.set_ylim(0, 70)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("0-DSP datapath:\n~order of magnitude\n(SoC overhead is the gap)", xy=(3, 5),
                xytext=(2.1, 30), fontsize=8, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    _save(fig, "ffn_block_energy.png")


if __name__ == "__main__":
    energy_per_token()
    activation_sparsity()
    fit_sweep()
    bandwidth_roofline()
    bram_fix()
    gather_savings()
    sparsity_compare()
    ffn_block_energy()
    print("done — 8 figures in", OUT)
