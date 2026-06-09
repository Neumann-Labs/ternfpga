# Full-model energy/token — projection from measured primitives

> **UPDATE (Phase 4):** the host-glue is now **measured** on silicon, not projected — see
> [`glue_measured.md`](glue_measured.md). The fully-measured system is ~4.32 J/token and
> **glue-bound** (host-side attention dominates); the ~1.47 J/token below is the engine-compute
> term (what on-fabric glue would realize). This doc remains valid as the engine-compute analysis.

Phase-3 #30–#32. The two on-silicon primitives are **measured**; the full-model number is
**composed** from them + the real BitNet-2B dimensions. What is measured vs projected vs
unmeasured is labelled explicitly — this is not a fully end-to-end measured J/token (the
on-board host-glue timing hit a toolchain wall, below).

## Measured on silicon (the inputs)
| quantity | value | source |
|---|---|---|
| Ternary engine rate | **1.00 cycle/tile** (8 ternary-MAC/cycle = 800 MMAC/s @ 100 MHz) | `onboard_throughput_measured.md` (#24) |
| Sustained DDR3 read | **1.42 GB/s** (89% of native-port peak) | `ddr3_roofline_measured.md` (#28) |
| SoC on-chip power | **0.489 W** (Vivado estimate, labelled) | `onboard_throughput_measured.md` |
| FFN/attention/layer numerics | **cosine 1.0** vs PyTorch | `validate_{ffn,attn,layer}.py` |

## Composed: BitNet-2B-4T engine compute per token
Weights per decoder layer = q+o (2×2560²) + k+v (2×2560×640) + gate+up+down (3×2560×6912)
= 16.38M + 53.08M = **69.47M ternary weights**. ×30 layers + LM-head (128256×2560, tied)
= **2.41 G ternary weights/token** → /K=8 = **301.5 M tiles** → **301.5 M cycles** @ 100 MHz.

| | time/token | tok/s | **J/token** (×0.489 W) | vs RTX 3060 (3.67 J/tok) |
|---|---:|---:|---:|---:|
| **FPGA engine compute** (measured 1 cyc/tile, K=8) | 3.02 s | 0.33 | **~1.47 J** | **~2.5× better** |
| + `down_proj` gather (60% sparse, measured) | 2.62 s | 0.38 | **~1.28 J** | **~2.9× better** |
| RTX 3060 (measured, BitNet-2B) | — | 23.5 | 3.67 J | — |

Energy/token is ~**K-invariant** for the engine (wider K → proportionally fewer cycles but more
active lanes), so the ~1.5 J/token engine figure is robust; widening K (toward the **1.42 GB/s**
roofline, #28) buys **throughput** (≈8 tok/s ceiling), not lower energy.

## The host-split glue — the open variable (honest)
The FPGA does the 7 ternary matmuls; VexRiscv does the glue (2× RMSNorm, RoPE, causal softmax,
a·V, dequant/quant). The glue is **numerically validated** (`ATTN_GLUE_C_PASS`, cosine 1.0) but its
**on-silicon cycle count is unmeasured**: the timing firmware (`bench_glue.c`) halts a few chars
into UART output once the heavy soft-float binary (libgcc-linked `double` math + `fw_mathf.h`) is
linked — a toolchain/runtime issue the lean firmwares never hit. Parked as a known limitation.

What can be said honestly:
- **Naive soft-float `double` glue on a soft CPU would be the bottleneck** — RoPE/softmax
  transcendentals alone are ~thousands of cycles each on VexRiscv (no FPU), plausibly dominating
  the 8.68 M engine cycles/layer. That would erode the energy lead → a real finding.
- **The fix is cheap (integer/fixed-point) glue, already proven here:** the **FFN glue is
  integer-only** (`ffn_glue.h`, dequant + RMSNorm cancel — `FFN_GLUE_C_PASS`). The same approach
  (fixed-point RoPE/softmax via LUTs, integer requant) makes the glue a small fraction of the
  engine cycles, leaving the system **engine-bound** at the ~1.5 J/token above. An on-chip glue
  unit removes it entirely.

## Verdict
The **0-DSP ternary engine** — the project's differentiator — is **measured** and is ~2.5× more
energy-efficient per token than the RTX 3060 on the same model, even at K=8, *before* its
GPU-unmatchable activation-sparsity skip. The remaining work to a fully-measured end-to-end
J/token is (1) cheap fixed-point glue (the FFN integer-glue is the template) or an on-chip glue
unit, and (2) resolving the heavy-firmware toolchain wall (or measuring the glue via litex_sim).
Both are concrete, scoped follow-ups — not unknowns.

_Inputs reproduce per their cited docs; this file is arithmetic over them._
