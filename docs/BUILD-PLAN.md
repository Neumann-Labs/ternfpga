# BUILD PLAN — A Ternary + Sparse LLM-Inference Engine on a $130 FPGA

**One sentence:** Build one hand-authored *multiplier-free, sparsity-skipping* ternary decode core on a single Arty A7-35T that, measured against the RTX 3060 in the same box, **wins energy-per-token, exploits per-token unstructured sparsity the GPU physically cannot, and doubles as the GPU's low-power speculative-decode draft engine** — conceding raw throughput on purpose.

> Detail lives in the four dossiers: `directions/A-ternary-engine.md`, `D-sparsity.md`, `B-wingman-specdecode.md`, `benchmark-methodology.md`. This is the synthesis + the phased plan.

## Why this is a real ML-Sys problem (not a toy)
Batch-1 LLM **decode** is **memory-bandwidth-bound** (~1–2 FLOP/byte) — the GPU's tensor cores sit idle while it streams weights. Two compounding escapes the field is actively chasing: **(1) ternary weights** (BitNet b1.58 → 1.58 bit/weight, multiply becomes sign-select, ~10× less traffic) and **(2) activation sparsity** (relu-fied/ProSparse FFNs are **85–95% zero per token**). A GPU **cannot natively do either**: no ternary datapath (it dequantizes to INT8/FP16) and no cheap *unstructured/dynamic* sparsity (only rigid 2:4). A purpose-built FPGA datapath does both. We compete on **joules/token and batch-1 latency**, the FPGA's home turf.

## The one core, three uses
A single reusable **ternary PE** = `acc += (w==+1 ? a : w==−1 ? −a : 0)` — a 6-LUT sign-select, **zero DSP in the multiply path** (all 90 DSP48 freed for norm/RoPE/softmax/quant). Wrap it three ways:
- **A** streams ternary weights from DDR3 → the energy/token engine.
- **D** gates which PEs fire from a per-token active-neuron mask → skips 85% of DDR3 traffic (the bandwidth bottleneck *is* the win).
- **B** runs a tiny on-chip ternary model as a **draft** proposing tokens the 3060 verifies → the GPU's wingman.

## Honest expected results (measured, not raw-throughput races)
| Claim | Number | vs 3060 |
|---|---|---|
| **A — energy/token** (same ternary model) | ~0.25–0.4 J/tok @ ~3 W | **~4–8× better** (concedes ~10–15× on tok/s) |
| **D — energy/token** (+85% activation sparsity) | ~0.07–0.13 J/tok; fetch ~15% of dense bytes | **~10–20× better**; ~85% FLOPs-skipped the GPU **can't** |
| **B — spec-decode** (async drafting hides 100 Mb-Eth) | ~1.5 J/tok, ~9 ms/tok | **~2.8× energy, ~2.7× latency** vs GPU-only |
| **Capability** | native ternary + per-token unstructured skip | **GPU has neither in silicon** |

*Anchors (all URL-verified): BitNet b1.58 (2402.17764), bitnet.cpp, TerEffic 455 tok/s/W (2502.16473), TeLLMe <7 W (2504.16266), ProSparse 89% sparsity (2402.13516), TEAL/Deja Vu/PowerInfer, batch-1 decode wall (2605.30571).*

## What fits (precise): a **~100–300M ternary, relu-fied model**
The 2B BitNet GGUF is 1.19 GB → **does not fit** 256 MB DDR3 (kept only as GPU baseline + RTL oracle). A ~300M ternary model ≈ 60 MB packed → fits with KV + headroom; FFN-heavy + ReLU/ProSparse so the 85% activation sparsity is real. DDR3 sustained ~0.6–0.8 GB/s is the throughput governor; sparsity directly relaxes it.

---

## Phased plan

### Phase 0 — De-risk the core (weeks 0–2) · **no FPGA/prod risk, no GPU needed → can start immediately**
- Stand up **bitnet.cpp** on worker4 CPU; reproduce tok/s + CPU-energy baseline on a small BitNet model (de-risks the model/tooling before any maintenance window).
- Hand-write the **ternary PE + 8×8 array** in SystemVerilog; **verilator + cocotb bit-exact** vs a NumPy ternary-matmul golden (zero mismatch); confirm **0 DSP** in the multiply path.
- Add the **sparse skip-decoder**; cocotb sim of DDR3-bytes-fetched **dense vs sparse** using a captured per-token active-index trace from a relu-fied model → the **bytes/FLOPs-skipped curve** (the §3.3 claim, simulated).
- First Vivado synth → LUT/DSP/BRAM/Fmax for one lane.
- **Decide the model:** distill/relu-fy from `1bitLLM/bitnet_b1_58-large` (0.7B) down to ~300M, or QAT a ~300M Transformer to ternary with **Brevitas**; export to the packed-sparse format the RTL consumes.

### Phase 1 — Real silicon, one layer, DDR3 + power (weeks 2–6)
- **LiteX SoC** on the Arty: DDR3 (LiteDRAM) + UART + Eth bring-up (the classic Arty time-sink, de-risked by LiteX); flash via openFPGALoader.
- Stream **one transformer block's** ternary weights from DDR3 through the array + sparse skip-decoder; measure **actual sustained DDR3 GB/s under gathered access** — the make-or-break number.
- **XADC self-power readback** (5 V current-sense → AUX ch 1/9) + inline USB meter → first **measured J/token + tok/s**, dense vs sparse, for one layer.
- **[Needs your coordination] GPU baseline:** schedule a maintenance window → install NVIDIA driver + CUDA on worker4 (prod-safe sequence in `benchmark-methodology.md` §4.1), verify Twenty CRM healthy, stand up llama.cpp-CUDA. *(Fallback: run the GPU baseline on a different CUDA box — the energy comparison stays valid since we measure the GPU's own power.)*

### Phase 2 — Full model + head-to-head + spec-decode (weeks 6–12)
- Full ~300M ternary relu-fied model decode loop (KV in DDR3, UART/Eth token I/O).
- **Apples-to-apples benchmark** (methodology §4): energy/token + tok/s + latency, Arty vs same model on 3060 (strict anchor) + realistic target on 3060; nvidia-smi long-run protocol (≥5 s, ≥4 trials) for ~5% power error; XADC/USB on FPGA.
- **D:** sparsity sweep (0/50/70/85/90%) → measured bytes/FLOPs/energy reduction.
- **B:** wire the Arty as draft model, 3060 as verifier; measure acceptance rate + coupled tok/s + total energy; validate the **async-continuation-drafting** fix for the Ethernet latency (Week-2 of this phase measures real RTT/p99 first).
- Populate the honest results tables + write-up.

## Prerequisites / decisions
1. **Model strategy** (Phase 0): distill-from-0.7B vs QAT-from-scratch a ~300M relu-fied ternary model. *Recommend: start with the public BitNet 2B as oracle + relu-fy/distill a ~300M; parallel-track a Brevitas QAT fallback.*
2. **GPU maintenance window** (Phase 1): worker4 runs prod Twenty CRM; the NVIDIA driver install needs a reboot. Needs your scheduling — or use a separate CUDA box.
3. **Toolchain:** ✅ Vivado 2025.2 + verilator + cocotb + openFPGALoader installed; board confirmed; 372 GB on `/srv/fpga`. (Minor: `vitis_hls` launcher PATH fix — only if we use HLS for the norm/softmax glue.)

## Immediate next step
**Phase 0 is safe to start now** — pure simulation + a CPU baseline, zero prod impact, no GPU/reboot needed — and it de-risks the single most important claim (multiply-free + sparse core, bit-exact, with the bandwidth-saving curve). The GPU window and model-training can run in parallel.
