# Scaling Feasibility — Ternary BitNet Decode on the Arty A7-35T

**Date:** 2026-06-08
**Method:** Multi-agent deep-research sweep — 105 agents, 23 primary sources fetched, 114
falsifiable claims extracted, 25 adversarially verified (3-vote, 2-of-3 refute to kill);
**22 confirmed, 3 refuted.** Sources are peer-reviewed/preprint primary papers (arXiv, MDPI).
**Question:** can we scale our working, 0-DSP, multiplier-free ternary GEMV core (proven
bit-exact on real silicon inside a LiteX VexRiscv SoC) toward full-model batch-1 LLM
*decode* on a $130 Arty A7-35T (~20,800 LUTs, 90 DSP48, ~225 KB BRAM, no URAM, single
DDR3 ~0.6–0.8 GB/s achievable, no hardened ARM)?

---

## Bottom line

1. **Our core technique is exactly right.** Three independent SOTA papers (TeLLMe v2,
   TerEffic, T-MAC) all converge on storing ternary `{−1,0,+1}` products in **LUT/distributed-RAM
   lookup tables, zero DSPs** — because pass/negate/zero selection wastes a DSP48's 25×18
   multiplier. The thing we hand-built is the field-standard method. ✅
2. **A *full* BitNet model does not fit a 35T.** The closest SOTA build (TeLLMe v2: full
   ternary BitNet b1.58 0.73B, 25 tok/s, 4.8 W) needs a **ternary core of ~23,082 LUTs —
   which alone exceeds our entire 20,800-LUT budget** — on a far larger Zynq part (KV260:
   ~256K LUTs, 1248 DSPs, hardened ARM, DDR4 17 GB/s, ~$250–350). Not transferable. ❌
3. **The binding constraint is memory bandwidth, not compute** — unanimous across every
   accelerator studied. Batch-1 decode streams the whole weight matrix once per token; our
   DDR3 is ~20–25× slower than even a KV260. We are locked into a **weight-streaming**
   regime; the high-throughput "all weights on-chip" regime is categorically unavailable
   (we have ~190× less on-chip memory than the U280 parts, and no URAM).
4. **The host-split architecture we chose is validated** — TeLLMe runs the non-ternary glue
   (RMSNorm, RoPE, online-softmax attention, SwiGLU) in fabric **and offloaded the LM head
   to its ARM CPU**. Our "VexRiscv handles glue / FPGA does matmuls" split is the right shape
   — but our host is a *soft* RISC-V core, much slower than the KV260's hardened A53, so the
   glue is a real latency risk.
5. **The activation-sparsity thesis is UNVERIFIED.** The single biggest claimed advantage in
   our own README (Direction D: 85–95% per-token sparsity → 10–20× energy/token) returned
   **zero supporting evidence** in the sweep. Must be measured ourselves before we bank on it.

**Honest, defensible scope on the 35T:** one **real-width transformer block** (or the ternary
GEMV kernel) streamed from DDR3, non-ternary glue + LM head on the VexRiscv host, with the
headline claim being **energy-per-token vs an RTX 3060 on bit-exact ternary numerics** — *not*
a full model, *not* throughput.

---

## 1. The 0-DSP LUT ternary core is validated SOTA  `[confidence: high]`

Ternary products `x·w ∈ {−x, 0, +x}`, so a DSP48 is "highly inefficient" — its multiplier is
wasted on a select/negate. Three primary sources independently build the lookup-table datapath
we built:

- **TeLLMe v2** (arXiv [2510.15926](https://arxiv.org/abs/2510.15926)): *"Using DSP48E1 blocks
  for 1.58-bit TLMM on FPGAs is highly inefficient… LUTs can efficiently handle pass, negate,
  or zero operations."* Ternary table-lookup matmul (TLMM) core = **23,082 LUTs, 0 DSP48**
  (stores all `3^G` group combinations, G=3 → 27 partial-sum entries).
- **TerEffic** (arXiv [2502.16473](https://arxiv.org/abs/2502.16473)): *"The whole TMat Core is
  composed of LUTs… leverages the advantages of ternary quantization"* — high weight bit = mux
  select, low bit = zero-gate; **~40% LUT savings** vs DSP-based accelerators (FlightLLM/EdgeLLM).
- **T-MAC** (arXiv [2407.00088](https://arxiv.org/abs/2407.00088)): bit-serial LUT kernels that
  *"scale linearly to the weight bit-width"* and eliminate multiplications.

> **Implication:** we are not on a wrong track. But the *scale* of TeLLMe's instantiation
> (23K LUTs for the ternary core alone) tells us the full-model version of this design is a
> KV260-class build, not a 35T build.

---

## 2. The bandwidth wall is the dominant constraint  `[confidence: high]`

Batch-1 decode arithmetic intensity ≈ 1 FLOP/byte → `tok/s ≈ usable_BW / bytes_per_token`,
where `bytes_per_token ≈ 0.2 × params` at 1.6 bit/weight. Confirmed by 8 claims across 4 sources:

- **TerEffic**: throughput collapses **22–24×**, from 16,300 tok/s (370M, *all weights on-chip*
  in URAM) to 727 tok/s (2.7B, from HBM), the moment weights exceed on-chip capacity. *"For a
  single-batch task, the HBM is not able to provide enough bandwidth to keep the compute core busy."*
- **FlightLLM** (arXiv [2401.03868](https://arxiv.org/abs/2401.03868)): naive decode reaches only
  **29–43% HBM utilization**; keeping activations on-chip raises it to 65.9%.
- **Embedded KV260** (arXiv [2502.10659](https://arxiv.org/abs/2502.10659)): 4-bit LLaMA2-7B at
  **~5 tok/s**, hitting 85% of the theoretical bandwidth limit on 19 GB/s DDR4.
- **Roofline cross-check** (TeLLMe): 680M decoder × ~0.2 B/weight ≈ 136 MB/token; 17.1 GB/s ÷
  136 MB ≈ 126 tok/s theoretical, **measured 25 tok/s ≈ 20% of peak** (DMA + KV overhead).

**On the Arty (~0.7 GB/s):** a 0.7B model would stream ~140 MB/token → **~5 tok/s theoretical
ceiling**, and a 7B model neither fits 256 MB DDR3 nor clears 1 tok/s. A **single block**
(~50M params at d_model≈2048) is ~10 MB/token → tens of tok/s, and fits DDR3 comfortably — the
streaming math only works at single-block / tiny-model scale.

---

## 3. The SOTA bar we're measured against  `[confidence: high]`

| System | Model | Board | Throughput (decode) | Power / efficiency | Notes |
|---|---|---|---|---|---|
| **TeLLMe v2** | ternary BitNet b1.58 **0.73B** | Kria KV260 (Zynq US+, ~256K LUT, 1248 DSP, ARM A53, DDR4 17 GB/s, ~$300) | **25 tok/s** (143 prefill) | **4.8 W** | Closest datapoint. Ternary core 23K LUT > our whole budget. LM head on ARM. |
| **PD-Swap** (Dec '25) | ternary BitNet | KV260 | **27 tok/s** | — | Marginally supersedes TeLLMe; still memory-bound. |
| **TerEffic** | ternary 370M / 2.7B | Alveo U280 (HBM, 42.6 MB on-chip) | 16,300 / 727 tok/s | (455 tok/s/W claim **refuted**) | Headline needs *all* weights on-chip — impossible on 35T. |
| **FlightLLM** | LLaMA2-**7B** (mixed ~3.5b/8b) | Versal VHK158 / U280 (HBM 460–819 GB/s) | **92.5 tok/s** | 4.2–6.0× energy eff. vs A100/V100S | Datacenter HBM parts, ~600–1000× our bandwidth. |
| **T-MAC** (CPU ref) | BitNet-b1.58-**3B** | Apple M2-Ultra | 30 (1-core) / 71 (8-core) tok/s | — | The CPU baseline framing for "what a tiny FPGA can't match on throughput." |

**White space:** *nobody in this corpus built an LLM on an Artix-7-class / sub-$150 board.* All
Arty-specific conclusions here are first-order roofline/resource extrapolations, not measured
results — directionally robust (the bandwidth wall and LUT ceiling hold across 13–1000× gaps)
but **empirically un-de-risked**. That gap is also the opportunity: a clean, reproducible
energy/token result on a $130 board is genuinely unoccupied.

---

## 4. The non-ternary datapath — host-split is validated  `[confidence: high]`

Ops that are **not** ternary matmuls and must coexist: RMSNorm (FP32 upcast), RoPE (precomputed
sinusoids in DDR), online-softmax attention (on-chip KV/score buffers), SwiGLU/SiLU elementwise,
residual adds, and the **LM head** (large final projection). In TeLLMe these consumed **610 DSPs
(49% of the KV260)** on the FP/INT path — *not* the ternary core — and the **LM head was offloaded
to the ARM CPU** (NEON W8A8, 9 ms) because fabric routing couldn't fit it.

> **Implication for us:** the architectural principle (split glue + LM head to the host) transfers
> and validates our design. **But** our host is a *soft* VexRiscv, not a hardened A53 — TeLLMe's
> 9 ms LM head would be far worse on VexRiscv, and FP softmax/RMSNorm/RoPE compete for our 90 DSPs.
> Host-side glue could dominate per-token latency and erode the FPGA matmul's energy win. This is
> the #2 feasibility risk and argues for a **tiny custom BitNet** (small LM head / hidden width).

---

## 5. On-chip residency does not transfer to a 35T  `[confidence: high]`

TerEffic's 16,300 tok/s headline comes *only* from keeping all ternary weights on-chip in **URAM**
(33.75 MB) + activations in BRAM (8.85 MB) = **42.6 MB** on a U280. The Arty A7-35T has **~225 KB
BRAM and zero URAM (~190× less)**. The high-throughput on-chip regime is categorically
unavailable; even small models force DDR3 streaming; careful BRAM tiling / double-buffering / DMA
feed is **mandatory but cannot escape the bandwidth ceiling** of §2.

---

## 6. CRITICAL GAP — activation sparsity is unverified  `[confidence: none — unanswered]`

The sweep returned **no evidence** on per-token unstructured activation sparsity in **BitNet b1.58
specifically**. We do not have a confirmed sparsity figure (the 85–95% hypothesis), nor evidence
that high sparsity is present in b1.58 vs. only in separate relu-fication lines (ProSparse,
Q-Sparse, relu-fied LLaMA). **Treat any sparsity-based throughput/energy multiplier as unverified
and high-risk.** This is the single largest open question and it directly underpins README
"Direction D." **It is also cheaply answerable by us** — instrument a CPU forward pass of the
BitNet model we already have and measure the real per-token zero rate. Do this before banking on it.

---

## 7. Realistic scope + strongest claim  `[confidence: medium]`

**Achievable & compelling on a 35T:** a **single transformer block at modest-to-real hidden width**
(or the ternary GEMV kernel), streamed from DDR3, with all non-ternary glue + LM head on the
VexRiscv host. **Not** a full BitNet 0.73B (ternary core alone > LUT budget), **not** 7B (won't
fit 256 MB, <1 tok/s).

**Strongest defensible claim:** energy-per-token vs an RTX 3060 on **bit-exact ternary numerics** —
perf/watt is the field-standard headline (FlightLLM headlines 4.2–6.0× vs A100/V100S; TeLLMe
headlines 25 tok/s @ 4.8 W). We concede throughput by design and compete where the GPU can't win.

### Top feasibility risks (could sink a multi-session build)

1. **DDR3 bandwidth** caps useful tok/s (§2) — unavoidable; size the model/block to it.
2. **LUT budget** — even one full-width ternary layer may overflow 20,800 LUTs → forces
   time-multiplexed / narrowed tiling. **Needs a direct synth/P&R fit experiment** (no source
   tested an Artix-7).
3. **Non-ternary glue on a soft VexRiscv** (§4) may dominate latency and erase the energy win →
   argues for a tiny custom BitNet.
4. **Activation-sparsity multiplier unproven** (§6) → measure before relying on it.

---

## Refuted claims (excluded from conclusions)

- TerEffic "455 tok/s/W, 192× a Jetson Orin Nano" edge-SOTA framing — **0-3 refuted**.
- "85% bandwidth limit ⇒ only ~15% datapath headroom" generalization — **0-3 refuted**.
- "370M needs 57.6 MB / two U280 cards" specific framing — **0-3 refuted** (the underlying
  on-chip-capacity-wall point survives via §5).

## Caveats

- **Part mismatch:** every primary source uses a part far larger/faster than the Arty (KV260,
  U280, VHK158, M2-Ultra). No measured Artix-7 LLM result exists — our numbers are roofline
  extrapolations.
- **Numeric precision:** the "0.6–0.8 GB/s" Arty figure is *achievable*, not peak (LiteDRAM
  default ~1.0, MIG ~1.3 GB/s peak); conclusions hold across the range. FlightLLM's "(INT4)" is
  actually ~3.5b-weight/8b-activation mixed precision.
- **Time-sensitivity (June 2026 vantage):** fast-moving field. TeLLMe's 25 tok/s already nudged
  by PD-Swap (27 tok/s, Dec 2025); newer ternary-specialized LUT work (Platinum, Vec-LUT) beats
  T-MAC's 2-bit interpretation ~1.3× — so T-MAC is the multiplier-free *scaling* reference, not
  the optimal ternary datapath.

## Primary sources

- TeLLMe v2 — end-to-end ternary BitNet on FPGA: https://arxiv.org/abs/2510.15926
- TerEffic — multiplier-free ternary accelerator: https://arxiv.org/abs/2502.16473
- FlightLLM — LLaMA2-7B on HBM FPGA: https://arxiv.org/abs/2401.03868
- T-MAC — LUT-based low-bit matmul (CPU): https://arxiv.org/abs/2407.00088
- Embedded-FPGA LLM bandwidth study (KV260): https://arxiv.org/abs/2502.10659
- PD-Swap (KV260, Dec 2025): https://arxiv.org/pdf/2512.11550
