# Phase 3 — full-model decode loop + a measured energy/token

_Grounded in the deep-research review (2026-06-08, 109 agents / 26 primary sources / 20
adversarially-verified claims). This is the spec; the loop builds against it._

## Why Phase 3 exists (the research verdict)

ternfpga today is **complete as a portfolio / engineering-blog artifact and a de-risking spike**:
the multiply-free ternary datapath is proven PyTorch (cosine 1.0) → sim (bit-exact) → silicon
(bit-exact), with a hardware-**measured** 1.00 cycle/tile. It is **not** complete as a publishable
paper or a product — and the gap is **specific and closeable**:

- The headline thesis is **energy-per-token vs a GPU**, but the strongest on-silicon number is
  **per-FFN-block**, derived — not a measured full-model decode loop.
- The field (FCCM/FPGA/FPL norm) wants **full-model, end-to-end, on-silicon, measured** results.
  The direct overlap, **TeLLMe v2** (arXiv 2510.15926, Oct 2025), runs a *full* BitNet 0.73B
  (prefill **and** decode) on a Kria KV260 at up to 143 tok/s prefill / 25 tok/s decode under ~5 W,
  measured. That is exactly the machinery we have not built.

What keeps the niche open (and worth finishing): **even TeLLMe does not cleanly report J/token**
(its energy figures are intelligence/J *ratios* — a clean-measured-J/token claim was *refuted* in
verification), the field's norm is GOPs/W not J/token, and our specific three-way intersection —
**ternary × per-token _unstructured_ activation sparsity × sub-$150 board** — is **genuinely
unoccupied** (every rival is 1.5–40× pricier, uses _structured_ sparsity, or is custom ASIC). The
**model** idea (ternary × sparsity) is _not_ ours to claim — Microsoft's BitNet a4.8 / Q-Sparse own
it — so our novelty is strictly **the cheap-hardware realization + the on-fabric unstructured
gather**.

## Phase-3 objective

> Produce a **measured full-model J/token** for a real ternary LLM on the **same $130 Arty A7-35T**,
> honestly framed, and characterize whether the energy advantage and the unstructured-sparsity
> differentiator survive at full-model scale.

That single result converts the project from "promising per-block datapath" to "credible end-to-end
energy claim" — the prerequisite for both publishability and for the headline being believable.

## Build decomposition (each increment committed + benchmarked)

| # | Task | Why | Tests |
|---|---|---|---|
| **#28** | **DMA weight feed + bandwidth roofline** | CPU CSR weight load won't scale to 0.7B; the decode loop needs weights streamed from DDR3. Widen K → bandwidth-bound. | **Risk 1** — measures real sustained DDR3 GB/s + the tok/s ceiling it implies. _Fail-fast._ |
| **#29** | **Attention datapath** (Q/K/V/O ternary GEMVs + softmax/RoPE/KV-cache glue) | The missing compute for a full layer. | Same PyTorch→sim→silicon chain as the FFN. |
| **#30** | **Single transformer layer on-board** (attn + FFN + 2×RMSNorm) | Integration milestone, measured cycles/layer, bit-exact. | End-to-end layer correctness. |
| **#31** | **Full decode loop** — embed → N layers (DDR3-streamed) → LM-head → sample, KV cache in DDR3 | The autoregressive machinery itself. | Real generated tokens. |
| **#32** | **Measured tok/s + J/token + head-to-head** | THE decisive deliverable. | The headline claim. |
| **#33** | **Full-model sparsity characterization** | Does ~60% stay _unstructured_ across all layers? | **Risk 2** — protects the differentiator. |

## The two risks this build is designed to expose (honestly)

1. **Single-channel DDR3 may cap the energy advantage at usable latency.** 0.7B ternary @ 2-bit ≈
   ~175 MB streamed per token; at ~700 MB/s that's ~250 ms/token → a low single-digit tok/s ceiling
   (TeLLMe's KV260 has faster DDR4 ~17 GB/s and still only hits 25 tok/s). The energy question is
   separate from speed — but if the number comes out badly, **#28 surfaces it in iteration 1–2, not
   after building attention.** We report whatever it is.
2. **The ~60% activation sparsity may not stay _unstructured_ at full-model scale.** If the per-token
   zero pattern is predictable enough that a structured N:M router (TENET-style) captures it, the
   unstructured-gather differentiator collapses. **#33** characterizes this before we write the
   novelty down.

## Model target

**BitNet b1.58 ~0.7B** — the largest that fits 256 MB DDR3 at 2-bit packing, and the same scale as
TeLLMe's 0.73B reference (clean comparison point). The full BitNet-2B (~600 MB packed) does **not**
fit on-board without weight paging from flash/SD — parked as future work. Exact dims are taken from
the actual model config during the build.

## Power methodology (decided)

**Vivado-estimated SoC power, clearly labeled as estimated** — the headline J/token = measured
cycles/token ÷ 100 MHz × Vivado-estimated power, same honest basis as the per-FFN-block number,
now extended end-to-end. **Live board-rail metering is deferred** (no rail instrumentation on hand
as of 2026-06-08); it is a clean future upgrade (USB inline meter / bench-PSU V×I / INA-class shunt)
that swaps the power term without touching the measured-cycles result.

## Definition of done (rounds the project out)

- A measured full-model **tok/s** and **J/token** on the $130 board (estimated-power basis, labeled).
- The DDR3 **roofline** (Risk 1) quantified — energy verdict stated whichever way it lands.
- The full-model **sparsity structure** (Risk 2) characterized.
- Head-to-head vs RTX 3060 + honest TeLLMe v2 / FlightLLM context; figures; BUILDLOG; README.
- _Then_ the remaining paper-grade gaps (live power, independent reproduction) are explicitly the
  named, parked next steps — not silent omissions.

_Re-sweep arXiv `cs.AR` for "ternary LLM FPGA" before writing any novelty claim down — the field
moves monthly (TeLLMe Oct '25, TOM Feb '26)._
