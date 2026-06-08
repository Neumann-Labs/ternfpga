# Direction D — Capturing the Sparsity GPUs Waste: An Activation-Sparse Ternary Decode Engine on the Arty A7-35T

**Thesis: batch-1 LLM decode is memory-bandwidth-bound, FFN activations are 40–95% zero per token, and a GPU is structurally forced to fetch and multiply those zeros anyway — so a tiny FPGA that fetches and computes *only the non-zero rows* can skip 40–90% of the DDR3 traffic that is the actual bottleneck, turning the Arty's greatest weakness (1.3 GB/s memory) into a win on energy-per-token and batch-1 latency that no dense GPU kernel can match.**

This is the sparsity arm of a three-part program. Direction A builds the ternary (1.58-bit) datapath; Direction B reuses it as a speculative-decode draft model; **Direction D makes that datapath *skip the work the GPU cannot skip.*** The ternary multiply-free PE *is* the sparse PE — we simply gate which PEs fire. The composition is the point: ternary shrinks each weight to ~1.6 bits, sparsity removes 40–90% of the weights you fetch at all, and the two savings *multiply*.

---

## 1. The problem & why the ML-Sys industry cares

Autoregressive decode generates one token at a time. Each step multiplies a single activation vector (batch 1) against the full weight matrices, so **every weight is read from memory and used exactly once** — arithmetic intensity is ~O(1) FLOP/byte, far below the ridge point of any modern accelerator's roofline. Decode is therefore **memory-bandwidth-bound, not compute-bound**: latency is set by how fast you can stream weights, and the GPU's TFLOPs sit idle. (Chen, *Memory-Bound but Not Bandwidth-Limited*, arXiv 2605.30571, makes exactly this characterization of the batch-1 wall and even shows faster HBM doesn't proportionally help because of launch-side overhead — a fixed cost the GPU pays that a streaming dataflow FPGA does not.)

The escape hatch the field discovered: **most of those weights don't matter for this particular token.** Two flavors:

**Activation sparsity (the big, dynamic, *unstructured* one).** In a gated FFN (`down_proj( act(gate_proj(x)) ⊙ up_proj(x) )`), if an intermediate activation is ~0, the entire corresponding column of `down_proj` and row of `up_proj`/`gate_proj` contribute nothing — they can be skipped *for this token*. The numbers are dramatic and well-replicated:

| Work | Model | Activation sparsity (zeros per token) | Quality | Source |
|---|---|---|---|---|
| ReLU Strikes Back (Apple) | OPT (ReLU) FFN | **>90%** per token, all layers | negligible loss | [arXiv 2310.04564](https://arxiv.org/abs/2310.04564) |
| ReLU Strikes Back | relu-fied Falcon-7B down-proj | ~1% → **95%** | minimal | same |
| ProSparse | **Llama-2-7B** | **89.32%** | ≈ Swish baseline | [arXiv 2402.13516](https://arxiv.org/abs/2402.13516) |
| ProSparse | Llama-2-13B / MiniCPM-1B | 88.80% / 87.89% | ≈ baseline | same |
| Deja Vu | OPT-175B (MLP + attn) | contextual sparsity **up to 80%** | no quality drop | [arXiv 2310.17157](https://arxiv.org/abs/2310.17157) |
| TEAL (training-free) | Llama-2/3, Mistral 7–70B | **40–50% model-wide** (incl. attention), **25% near-zero loss** | minimal at 40% | [arXiv 2408.14690](https://arxiv.org/abs/2408.14690) |
| CATS | Mistral-7B / Llama-7B (SiLU) | **50%** controllable | within 1–2% | [arXiv 2404.08763](https://arxiv.org/abs/2404.08763) |
| PowerInfer | OPT-30B / Llama2-70B | **17–43%** of neurons = 80% of activations (power-law) | — | [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) |

The industry cares because this directly buys decode speed. TEAL reports **1.53× / 1.8×** wall-clock decode speedup at 40% / 50% sparsity, and crucially states the *why*: **"weight channels associated with zero-valued activations are unnecessary during decoding, and we can achieve speedup by avoiding the transfer of such channels to on-chip memory."** Deja Vu gets **>2×** on OPT-175B with a learned predictor; PowerInfer gets **7.23× average** (up to 11.69×) over llama.cpp by routing hot neurons to GPU and cold to CPU; ProSparse reports up to **4.52×**. The mechanism is *memory-traffic reduction*, which is exactly what a bandwidth-starved device needs.

**Unstructured weight sparsity** (one-shot pruning, e.g. SparseGPT/Wanda-style) is the static cousin: 50–60% of weights can often be zeroed with little loss, but the surviving non-zeros land at *arbitrary* positions.

**Why this is the right problem for *this* hardware.** The Arty loses raw throughput to the 3060 by ~280× on bandwidth (1.3 GB/s vs ~360 GB/s). But if 90% of FFN weight traffic is provably useless this token, the *effective* gap shrinks toward ~28×, and on **energy-per-token** and **batch-1 latency** — where the FPGA's tiny fixed power (a few watts) and zero kernel-launch overhead dominate — the FPGA can win outright.

## 2. The core idea & the FPGA edge (what the GPU structurally CANNOT do)

**The skip we exploit is per-token, data-dependent, and unstructured.** Which FFN neurons are active changes *every token* and follows no fixed pattern. A GPU's tensor cores are SIMD/SIMT: a warp executes the same instruction across 32 lanes, and the dense GEMM/GEMV kernels read contiguous weight tiles. To get speedup, the GPU needs the *non-zeros packed into rigid, hardware-blessed structure* — NVIDIA Ampere/Ada sparse tensor cores accelerate **only 2:4 structured sparsity** (exactly 2 non-zeros per aligned group of 4), a fixed *static weight* pattern decided at train/prune time. There is **no NVIDIA hardware path** that accelerates "this token, these 1,113 of 11,008 FFN neurons are non-zero, gather their columns." On a GPU, dynamic per-token activation sparsity either (a) runs as a dense GEMV that multiplies the zeros anyway, or (b) attempts gather kernels whose irregular memory access and warp divergence usually *erase* the savings below ~90% sparsity. That is the structural waste.

**What the FPGA does that the GPU can't:**

1. **Fine-grained predication at the row level, for free.** On the FPGA we build the weight-fetch address generator and the ternary PE array ourselves. A 1-bit "active" mask per FFN neuron directly gates *whether we issue the DDR3 burst for that neuron's weight column and whether the PEs fire.* There is no warp, no 32-lane minimum, no fixed 2:4 group — granularity is **1 neuron**. Skipped work costs *zero cycles and zero DRAM reads* instead of being multiplied-by-zero.
2. **Custom gather address generation overlapped with compute.** We stream a *compressed list of active neuron indices* and let a dedicated address-generation FSM prefetch the corresponding DDR3 rows while the PE array consumes the previous batch — hiding irregular-access latency behind compute, which a fixed GPU memory hierarchy cannot reconfigure to do.
3. **Sparsity × ternary compose natively.** Each surviving weight is a ternary {−1,0,+1} packed at ~1.6 bits (à la TerEffic's 5-values-in-8-bits encoding). So the bytes we *do* fetch are ~10× smaller than FP16 *and* we fetch 2–10× fewer of them. The PE is an add/subtract/skip — no multiplier — so an "active" neuron costs a few LUTs, and the DSP48s are freed for the predictor and accumulation.

**The honest catch (addressed in §3/§7):** gather/scatter and irregular DDR3 access are *exactly* the things FPGAs are also bad at. SpMV literature is unanimous that the hard part is "input-dependent memory access patterns" and the low FLOP/byte of reading the vector via indices ([Serpens, arXiv 2111.12555](https://arxiv.org/pdf/2111.12555)). Our design wins only because (a) we apply sparsity to the *weight* dimension where the access, though gathered, is still a contiguous burst *per neuron* (not per element), and (b) the predictor lets us gather **ahead of time** rather than discovering zeros mid-multiply. We quantify the overhead, not wish it away.

**Precedent that the gap is real and unfilled.** TerEffic (the ternary-FPGA anchor, arXiv 2502.16473) explicitly states it focuses on ternary quantization and **does *not* exploit sparsity**, noting that FlightLLM is the design that "leveraged sparsity." FlightLLM (arXiv 2401.03868) built a *configurable sparse DSP chain* on a big Alveo U280 and hit 55 tok/s on Llama2-7B batch-1 with up to **6.0× better energy efficiency** than the GPU. **Nobody has combined ternary + per-token activation sparsity on a sub-$150 board.** That intersection is Direction D.

## 3. Technical design on the Arty A7-35T

### 3.0 Resource budget (the hard ceiling)
Arty A7-35T (XC7A35T): **20,800 LUTs, 41,600 FFs, 90 DSP48E1, 1.8 Mb (~225 KB) BRAM** (50× RAMB36), 256 MB DDR3L 16-bit @ 667 MHz DDR. Theoretical DRAM peak = 1333 MT/s × 2 B = **2.67 GB/s**; realistic sustained via MIG with our access pattern **~0.6–0.9 GB/s** (we budget 0.7). This number dominates everything below.

### 3.1 What model FITS (be precise)
A 7B model does not remotely fit and is not the target. The right scale:
- **Primary: a ternary ~150–300M FFN-heavy model** (relu-fied / ProSparse-style so activation sparsity is intrinsic and high). 300M ternary params ≈ 300M × 1.6 bits ≈ **60 MB** in DDR3 — fits with KV-cache and headroom in 256 MB.
- **Stretch: ~700M–1B ternary** (≈ 140–200 MB), KV-cache permitting; feasible but bandwidth-tighter.
- **Train/convert a custom relu-fied small model** so we *own* the sparsity (a 7B-class model can't be the workload, but a relu-fied/ProSparse 0.3–1B can — and that is the published sweet spot: ProSparse MiniCPM-1B hits 87.9% sparsity).

We deliberately pick a model whose FFN dominates parameter count and is **ReLU/ProSparse-sparsifiable**, because that is where the 90% skip lives and where DDR3 traffic is highest.

### 3.2 Dataflow architecture (decode, batch 1)

```
                          ┌──────────────────────────────────────────────┐
   token x (d_model, INT8)│                 Arty A7-35T                   │
        │                 │                                              │
        ▼                 │   ┌────────────┐    active-neuron index list │
  ┌───────────┐           │   │ PREDICTOR  │───►(compressed, on-chip)────┐│
  │ on-chip   │           │   │ (low-rank/ │                            ││
  │ x, resid  │──────────►│   │  thresh)   │       ┌────────────────────▼┴─┐
  │  (BRAM)   │           │   └────────────┘       │ GATHER ADDR-GEN FSM    │
  └───────────┘           │         ▲              │ idx → DDR3 burst addr  │
        ▲                 │         │ x            │ (prefetch, double-buf) │
        │ y               │   ┌─────┴───────────┐  └───────────┬───────────┘
  ┌───────────┐           │   │ TERNARY PE ARRAY│              │ ternary weight rows
  │ accumulate│◄──────────│   │  add/sub/skip   │◄─────────────┤ (only ACTIVE neurons)
  │  (DSP+BRAM)           │   │  (LUT-based)    │              │
  └───────────┘           │   └─────────────────┘    ┌─────────▼─────────┐
                          │                          │  DDR3 (MIG)        │
                          │                          │  ternary weights   │
                          └──────────────────────────│  + KV cache (256MB)│
                                                      └────────────────────┘
```

**Per-layer decode pipeline:**
1. **Predict active set.** For the FFN, compute the gate projection (or a cheap low-rank predictor of it, Deja-Vu style) to get the `gate_proj(x)` activations; apply a **contextual threshold** (CATS/TEAL-style magnitude threshold, or ReLU's exact zero test) → a 1-bit mask + a packed list of active neuron indices. Deja-Vu/PowerInfer predictors run >95% recall at ~6% param overhead; we can also use the *exact* gate result (no predictor, no mis-prediction) since gate_proj itself is ternary and cheap — a design knob.
2. **Gather + stream only active rows.** The addr-gen FSM converts active indices into DDR3 burst addresses for `up_proj`/`down_proj` columns, double-buffered so DRAM bursts for batch *k+1* overlap PE compute on batch *k*.
3. **Ternary MAC, no multiplier.** PE array does signed accumulate (+w·x with w∈{−1,0,+1} ⇒ add/sub/skip). With 90 DSP48 free, DSPs do attention/accumulation and the predictor's small dense matmul; the bulk FFN is LUT-based add/sub. Target ~256 PE lanes, ~100 MHz.
4. **Scatter-accumulate** active contributions into the output (down_proj) accumulator in BRAM; write residual back on-chip.

**On-chip vs DDR3:** activations, residual stream, KV for the current step, the active-index list, and the predictor weights live in **BRAM** (225 KB is enough for d_model≤2048 activations + a few KB index list). The **ternary weight matrices live in DDR3** and are streamed; *this stream is what sparsity shrinks.* KV cache also in DDR3.

### 3.3 The bandwidth math (the whole ballgame)
For a 300M ternary FFN-heavy model, per-token *dense* weight traffic ≈ 60 MB. At 0.7 GB/s that's ~86 ms/token → ~12 tok/s dense. **Apply 85% activation sparsity:** fetch ~9 MB/token → ~13 ms → **~75 tok/s** on FFN-bound layers (attention adds overhead). The point isn't the absolute number — it's that **sparsity converts the DDR3 bottleneck from the wall into a 5–7× lever**, and it stacks on ternary's byte-shrink.

### 3.4 Hand-RTL vs Vitis HLS split
- **Hand-written SystemVerilog (verilator/cocotb-tested):** the ternary PE array, the gather **address-generation FSM + double-buffer controller** (latency-critical, irregular — HLS struggles here), and the mask/index packing unit. These are the parts where cycle-exact control of DDR3 bursts wins or loses the bandwidth game.
- **Vitis HLS:** the predictor matmul (small dense, regular), the threshold/top-k unit, layernorm, and attention softmax — dataflow-friendly, fast to iterate.
- **MIG (Memory Interface Generator):** DDR3 controller (don't hand-roll).

## 4. Benchmark vs the RTX 3060 — exact methodology

**Apples-to-apples principle:** *same model weights, same prompt set, same output tokens, measure energy and latency at the wall.* The FPGA's only "cheat" is that it skips zeros — which the GPU is *also allowed* to skip if it can, so we run the GPU both dense and with the best available sparse path.

**Reference model & quantization.** A relu-fied/ProSparse small model (~0.3–1B) in **two forms**: (a) ternary for the FPGA, (b) the same model at **INT8/INT4 (e.g. AWQ)** on the 3060 via a standard runtime (llama.cpp/vLLM). We also run the GPU with **TEAL/CATS activation-sparse kernels** to give the GPU its *best* sparse shot — and show it still can't match the FPGA's fine-grained skip on energy.

**Metrics:**
- **Energy per token (J/tok)** — primary headline.
- **Batch-1 decode latency (ms/tok, p50/p99)** — secondary headline.
- **Throughput (tok/s)** — reported honestly; FPGA expected to lose absolute tok/s.
- **FLOPs-skipped (%)** and **DRAM-bytes-skipped (%)** — the capability metric proving the GPU left work on the table.

**Power measurement — FPGA:** Arty lacks rich on-board telemetry, so use an **inline USB/DC power meter** on the 5 V/12 V barrel (or a bench PSU with current readout) sampling ≥1 kHz; integrate over a fixed N-token decode to get J/tok. Cross-check against Vivado's post-implementation **XPE/power report** for the static+dynamic estimate. Report wall-plug device power (the honest number includes the FPGA's idle draw).

**Power measurement — GPU:** `nvidia-smi --query-gpu=power.draw --format=csv -lms 50` integrated over the *same* N-token run, plus a wall-meter on the GPU rail for cross-check. Subtract idle to report *incremental* decode energy, and also report total — both, so neither side is flattered.

**Controls:** identical tokenizer, identical 100–500 prompt suite, fixed `max_new_tokens`, greedy decode, warm cache, ≥10 runs, report mean ± std. Log the **measured per-token activation-sparsity histogram** on both so the FLOPs-skipped claim is grounded in the actual run, not the paper.

## 5. Honest expected numbers

Assumptions: ~300M relu-fied/ProSparse ternary model; **85% mean FFN activation sparsity** (conservative vs ProSparse's 89%); Arty sustained DDR3 **0.7 GB/s**; FPGA wall power **~4–6 W** (board + DDR3); 3060 idle ~15 W, decode ~110–130 W; GPU running INT4 dense at batch 1.

| Metric | RTX 3060 (INT4 dense, batch 1) | Arty A7-35T (ternary + 85% act-sparse) | Verdict |
|---|---|---|---|
| Throughput (tok/s) | ~60–120 | **~40–75** | **GPU wins** ~1.5–2× (honest loss) |
| Batch-1 latency p50 (ms/tok) | ~8–16 (+ launch overhead) | **~13–25** | ~toss-up; FPGA competitive, no launch overhead |
| Device power (W, decode) | ~110–130 | **~5** | **FPGA ~25×** lower |
| **Energy/token (J/tok)** | ~1.0–2.0 | **~0.07–0.13** | **FPGA ~10–20× better** ← headline |
| DRAM bytes/token | full dense fetch | **~15% of dense** | capability GPU lacks |
| FLOPs **skipped** by sparsity | ~0% (dense kernel) / partial w/ TEAL | **~85%** at 1-neuron granularity | structural GPU gap |
| Native unstructured/per-token skip | **No** (needs 2:4) | **Yes** | the moat |

**What wins:** energy-per-token (~10–20×), DRAM-traffic-skipped (~6–7×), and *capability* (per-token unstructured skip the GPU can't do at all). **What loses:** raw throughput (~1.5–2×) and possibly p99 latency under predictor mispredict. **The sparsity *delta* on the FPGA alone:** ternary-dense ≈ 12 tok/s → ternary+sparse ≈ 40–75 tok/s, i.e. sparsity is a **~4–6× lever** on top of ternary, *net of* predictor + gather overhead (~10–15% of cycles, modeled on Deja-Vu's <10% predictor time and our prefetch hiding the gather).

## 6. Milestones

**2-week (prove the lever in simulation).**
- Cocotb/verilator model of the **gather addr-gen FSM + ternary PE lane**; feed it a captured per-token active-index trace from a relu-fied 300M model (extracted in PyTorch).
- Cycle-accurate sim of DRAM-bytes-fetched dense vs sparse → produce the **FLOPs-/bytes-skipped curve vs sparsity (0–95%)**. Deliverable: the §3.3 bandwidth claim, simulated.

**6-week (single sparse FFN layer on real silicon).**
- One FFN layer end-to-end on the Arty: MIG DDR3 + predictor (HLS) + mask/index pack + gather FSM + ternary PE array (RTL). Measure **real sustained DDR3 GB/s** under the gathered access pattern (the make-or-break number).
- Inline power meter rig; report J per FFN-layer-token dense vs sparse. Deliverable: measured sparsity speedup + energy on hardware for one layer.

**12-week (full small model + GPU benchmark).**
- Full ~300M ternary relu-fied model decoding on the Arty (layers streamed from DDR3); integrate Direction A's ternary datapath.
- Run the §4 head-to-head vs the 3060 (energy/token, latency, tok/s, measured sparsity histogram). Deliverable: the §5 table populated with *measured* numbers + write-up; optional Direction-B hook (use it as the 3060's draft model).

## 7. Risks & mitigations

- **Biggest "this won't work because…": gather/scatter irregularity will eat the savings (the SpMV curse).** *Answer:* we sparsify the **weight-fetch (column) dimension**, where each active neuron is still a **contiguous DDR3 burst**, not a per-element scatter — so we never do single-word random reads. The predictor gives indices *ahead of time*, so the addr-gen FSM **prefetches and double-buffers**, hiding gather latency behind PE compute. We only claim a win above the crossover sparsity (~50–60%) where saved bursts dominate index overhead — and relu-fied FFNs sit at 85–90%, comfortably above it. Below 50% sparsity we fall back to dense streaming (a runtime mode switch), so we never lose to ourselves.
- **DDR3 sustained bandwidth < 0.7 GB/s under gathered access.** *Mitigation:* this is *the* 6-week gate; if MIG can't sustain it, increase burst length per neuron (fetch full up+down columns contiguously), pin hot neurons (PowerInfer-style) in BRAM to cut DRAM trips, and shrink the model. The 2-week sim de-risks the target before silicon.
- **Predictor mispredicts → quality drop or recompute.** *Mitigation:* use the **exact** ReLU/threshold gate result (no learned predictor, zero false negatives) for the primary design — gate_proj is ternary and cheap on-chip; reserve the learned low-rank predictor (Deja-Vu, >95% recall) only for the stretch model. Validate end-task accuracy vs dense at each sparsity.
- **Control logic blows the 20,800-LUT budget.** *Mitigation:* ternary PEs are LUT-cheap (add/sub/skip), DSPs carry the dense predictor/attention; the index FSM is small. Time-multiplex PE lanes if needed — we're bandwidth-bound, not compute-bound, so fewer lanes is fine.
- **"Why not just buy a bigger FPGA / use the GPU's 2:4?"** *Answer:* the thesis is *not* to beat the GPU on throughput; it's energy-per-token and a *capability* (per-token unstructured skip) that no 2:4 path provides — demonstrated on a $130 board, which is the whole proof-of-concept's value.

## 8. Key resources (verified URLs)

- Deja Vu — contextual sparsity, up to 80%, learned predictor, >2× on OPT-175B — https://arxiv.org/abs/2310.17157 · code https://github.com/FMInference/DejaVu
- ReLU Strikes Back (Apple) — >90% FFN sparsity, aggregated sparsity — https://arxiv.org/abs/2310.04564
- ProSparse — Llama-2-7B **89.32%** activation sparsity, up to 4.52× — https://arxiv.org/abs/2402.13516
- TEAL — training-free 40–50% model-wide, 1.53×/1.8×, "avoid transferring zero-activation channels" — https://arxiv.org/abs/2408.14690 · code https://github.com/FasterDecoding/TEAL
- CATS — contextual thresholding, 50% controllable, within 1–2% — https://arxiv.org/abs/2404.08763
- PowerInfer — hot/cold neurons, power-law, >95% predictor, 7.23× over llama.cpp — https://arxiv.org/abs/2312.12456
- TerEffic — ternary FPGA anchor; **explicitly no sparsity** (the gap D fills) — https://arxiv.org/html/2502.16473
- FlightLLM — configurable sparse DSP chain on FPGA, 6.0× energy eff. — https://arxiv.org/abs/2401.03868
- Batch-1 decode wall — memory-bound characterization — https://arxiv.org/abs/2605.30571
- Serpens — FPGA SpMV, the gather/irregular-access reality check — https://arxiv.org/pdf/2111.12555
- Arty A7 reference (DDR3 256 MB 16-bit 667 MHz, MIG) — https://digilent.com/reference/programmable-logic/arty-a7/reference-manual
