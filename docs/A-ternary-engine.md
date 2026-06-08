# Direction A — A Multiplier-Free Ternary GEMV Engine on a $130 FPGA

**Thesis.** A BitNet b1.58 (weights ∈ {−1, 0, +1}) decode datapath on one Arty A7-35T turns every matrix-vector multiply into LUT-based sign-selects and adder trees — *zero DSP multipliers in the hot loop* — so the board runs a ~300M-parameter ternary LLM at single-digit tokens/sec on ~3 W. It will **lose** raw throughput to an RTX 3060 by a wide margin, but it should win **energy-per-token by ~20–40×** and own a capability the GPU does not have in silicon: native ternary arithmetic with no dequantization.

---

## 1. The problem & why the ML-Sys industry cares

Autoregressive **decode** (generating one token at a time, batch-1) is the dominant cost of interactive LLM serving, and it is **memory-bandwidth-bound, not compute-bound**. Every generated token requires streaming the *entire* weight matrix from memory through the compute units exactly once; the arithmetic intensity is ~1–2 FLOP/byte, far below the "ridge point" of any modern GPU. A 7B model in FP16 is ~14 GB of weight traffic per token; even at 360 GB/s that is a hard floor of ~26 ms/token *before any compute*. The recent "decode wall" analysis ([arXiv:2605.30571](https://arxiv.org/abs/2605.30571), *"Memory-Bound but Not Bandwidth-Limited"*) makes this sharper: it measures an **L4 GPU hitting ~81% of its bandwidth floor while an H100 reaches only ~27%** on the same batch-1 workload — i.e., faster memory does not buy proportional latency because launch-side overhead is exposed. The takeaway for hardware designers: in the decode regime, **the model that moves the fewest bytes per token, at the lowest energy per byte, wins** — and tensor-core FLOPS are largely wasted.

Two structural facts make this a real opening, not a toy:

1. **Ternary weights collapse the byte budget.** Microsoft's **BitNet b1.58** ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) showed that weights quantized to {−1, 0, +1} (log₂3 ≈ **1.58 bits/weight**) match FP16 perplexity from ~3B params upward. The released **BitNet b1.58-2B-4T** ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285), [HF model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)) reports a **0.4 GB non-embedding footprint** and an estimated **0.028 J/token** on a laptop CPU — a 1.58-bit weight is ~10× less traffic than FP16. The decode wall and ternary weights compose: cut bytes/token by 10× and the bandwidth floor drops 10×.

2. **GPUs cannot natively multiply by a ternary weight.** A tensor core's MAC array is built for INT8/FP16 products. To run BitNet on a GPU you **dequantize** ternary → INT8/FP16 in registers, then feed the normal MAC array — you pay full-width memory traffic *unless* you hand-pack, and you never exploit that the "multiply" is really a sign-select. This is exactly the gap the LUT-based mixed-precision GEMM line of work attacks in *software*: **T-MAC** ([arXiv:2407.00088](https://arxiv.org/abs/2407.00088), [microsoft/T-MAC](https://github.com/microsoft/T-MAC)) precomputes partial sums into lookup tables and replaces mpGEMM with table lookups + shift-accumulate, reporting up to **4× throughput and ~70% energy reduction** on CPUs and beating GPU/NPU in single-batch decode. **bitnet.cpp** ([microsoft/BitNet](https://github.com/microsoft/BitNet)) reports **1.37–5.07× speedups and 55–70% energy reduction on ARM, 2.37–6.17× / 72–82% on x86**. These are *CPU* wins from doing ternary correctly. An FPGA can bake that LUT/sign-select datapath directly into the fabric.

**Why ML-Sys cares:** perf/watt and batch-1 latency are the metrics that decide on-device and edge deployment, draft-model economics for speculative decoding, and serving cost at the margin. A credible "ternary done in hardware" result is a data point in an active research front (TerEffic, TeLLMe, TENET, FlightLLM, LUT Tensor Core — all 2024–2025).

---

## 2. The core idea & the FPGA edge

**The ternary multiply is free.** For weight *w ∈ {−1, 0, +1}* and 8-bit activation *a*:

```
w = +1  →  +a
w =  0  →   0
w = −1  →  −a
```

This is a 2-bit-control sign/zero select — a single 6-input LUT, **no DSP, no carry**. A GEMV row reduces to: gate each activation by its weight's sign, then sum. The whole engine is **adder trees + a streaming LUT-packed weight feed**; the 90 DSP48E1 blocks are *freed* from the multiply and reused for what they are good at — activation quantization (absmax scaling), accumulator widening, and the few true multiplies (RMSNorm/RoPE/softmax scale).

**What the GPU structurally cannot do that the FPGA can:**

| Capability | RTX 3060 (GA106) | Arty A7-35T |
|---|---|---|
| Native ternary MAC (no dequant) | No — tensor cores are INT8/FP16; ternary is emulated/dequantized | **Yes** — sign-select LUT is the literal datapath |
| Pack weights at exactly 1.58 b in the compute path | No — operands widen to ≥INT8 at the ALU | **Yes** — 5 ternary weights → 1 byte fed straight to LUTs |
| Spend zero multiplier area on the dominant op | No — the MAC array *is* multipliers | **Yes** — 90 DSPs left for quant/accumulate/norm |
| Datapath shaped to ~1.6 bit/weight bandwidth | Memory controller + caches sized for wide types | **Yes** — DDR3 burst + on-chip layout co-designed for ternary |

The published FPGA results validate the direction: **TeLLMe** ([arXiv:2504.16266](https://arxiv.org/abs/2504.16266)) implements a *table-lookup* ternary matmul engine on a Kria KV260 that encodes groups of 3 ternary weights into a 5-bit index over 3³ = 27 partial-sum combinations, using **52,094 LUTs vs 59,999 for a naïve add/subtract** array — and runs a 0.7B model at **9.51 tok/s under 7 W**. **TerEffic** ([arXiv:2502.16473](https://arxiv.org/abs/2502.16473)) reports a 370M model fully on-chip at **16,300 tok/s and 455 tok/s/W**, and a 2.7B HBM-assisted design at **727 tok/s, 46 W, 16 tok/s/W**, claiming **19× efficiency vs a Jetson Orin Nano** (370M) and **8× energy efficiency vs an A100** (2.7B). The Arty is far smaller than either board, so I target their *energy efficiency* shape at a *tiny* scale, never their throughput.

---

## 3. Technical design on the Arty A7-35T

### 3.1 The hardware budget (the honest constraints)

| Resource | Arty A7-35T (XC7A35T) | Consequence for this design |
|---|---|---|
| LUTs | 20,800 (6-input) | Sign-select array + adder trees must be *tiled* and time-multiplexed; cannot unroll a full layer |
| DSP48E1 | 90 | Reserved for quant scale, accumulator add, RoPE/softmax — **0 used for ternary multiply** |
| BRAM | 1.8 Mb = **~225 KB** | Holds activations + one streamed weight tile + KV for *small* layers — **weights cannot live on-chip** |
| DDR3L | 256 MB, MT41K128M16, x16 | **Weights stream from here every token.** This is the throughput governor. |
| DDR3 bandwidth | x16 @ DDR3L-1600 → 3.2 GB/s peak theoretical; **MIG on Arty realistically ~325–667 MHz → ~1.3 GB/s peak, ~0.5–0.8 GB/s sustained** | The bytes/token × this number = the token rate ceiling |
| Host link | 100 Mb Ethernet + USB-UART, **no PCIe** | Prompt/token I/O only; not in the inner loop |
| Board power | XC7A35T <0.5 W typical real design (20 mW static); **board total ~2–4 W** incl. DDR3 + Ethernet PHY | The denominator of energy/token |

**The single most important honesty point:** with only ~225 KB of BRAM, the TerEffic "all weights on-chip" strategy is **impossible** on the Arty. Weights *must* stream from DDR3 on every decode step, so **DDR3 bandwidth is the hard ceiling** and the design is a DDR3-streaming engine, not an on-chip-SRAM engine.

### 3.2 What model fits (be precise)

- **BitNet b1.58-2B-4T does NOT fit.** Its GGUF `i2_s` file is **1.19 GB** ([HF gguf repo](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)) — ~5× the 256 MB DDR3. Even ignoring activations/KV it overflows. *Do not claim the 2B runs on this board.*
- **Target: a ~150–350M ternary BitNet-class model.** A 300M-param model at 1.6 bits/weight ≈ **60 MB** of weights — comfortably inside 256 MB with room for the 128k-entry embedding (kept FP16/INT8 in DDR3, ~64–128 MB if full; better to use a *smaller-vocab* trained model or 8-bit embeddings), KV cache, and activation scratch. Concretely:
  - **Option A (recommended):** train/distill a **~300M BitNet b1.58** (e.g., 24 layers, hidden 1024, FFN 2730, GQA) using the public BitNet training recipe / Microsoft BitNet repo. Owns the IP, sized for the board, ~60 MB weights.
  - **Option B (off-the-shelf demo):** use the released **2B config** ([config.json](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T): 30 layers, hidden 2560, FFN 6912, 20 heads / 5 KV heads, head_dim 128, vocab 128,256) **only as the RTL correctness oracle / GPU baseline**, and run the *small* model on-board. The 2B is the numerical golden reference; the on-board model is the ~300M.
- **KV cache:** at hidden 1024, GQA with 4 KV heads × head_dim 128 → ~1 KB/token/layer in INT8; 24 layers × 4096 ctx ≈ **~100 MB** worst case — keep KV in DDR3, stream per layer. For short contexts (≤512) KV fits more easily; design assumes KV in DDR3 with a small BRAM working window.

### 3.3 The exact datapath (decode, one token)

```
DDR3 (weights @1.58b, KV @INT8, embeddings)
   │  burst read, weight tile = N rows × K cols, packed 5 ternary/byte
   ▼
[Weight unpacker]  byte → 5 × 2-bit ternary codes (LUT ROM)         ← no DSP
   │
   ▼
[Sign-select array]  for each (w, a):  +a / 0 / −a                  ← 6-LUT, no DSP
   │   (activations a held in BRAM, INT8, per-token absmax scaled)
   ▼
[Adder reduction tree]  sum K products → partial accumulator        ← LUT/CARRY4
   │   wide accum (INT24+) optionally widened in DSP48 ALU mode      ← DSP (accum only)
   ▼
[Activation quantizer]  absmax → INT8, scale stored                 ← DSP (scale mult)
   │   RMSNorm, RoPE, squared-ReLU FFN, softmax  (BitNet b1.58 ops)  ← DSP for the few real mults
   ▼
[Output activation buffer (BRAM)]  → feeds next layer / KV write-back
```

**Ternary multiply as LUT, concretely.** Two encoding choices, both DSP-free:

- **Direct sign-select (simplest):** pack 5 ternary weights into 1 byte (3⁵ = 243 ≤ 256). A small ROM expands byte → 5 control pairs; each pair drives a select on the INT8 activation (`+a`, `0`, `−a`). Adder tree sums. This is the cleanest RTL and the recommended v1.
- **Table-lookup partial sums (TeLLMe-style, denser):** group **G = 3** weights, precompute all 3³ = 27 partial sums `Σ wᵢ·aᵢ` for the current activation triple into a 27-entry table, then index with the 5-bit weight code. TeLLMe shows this saves ~13% LUTs vs naïve add/subtract at G=3, T=32 parallel tables ([arXiv:2504.16266](https://arxiv.org/abs/2504.16266)). Adopt for v2 once the direct version is correct.

**Parallelism on 20,800 LUTs.** Budget ~8–16 dot-product lanes, each handling a K-tile of the GEMV with a pipelined adder tree, time-multiplexed across the N output rows of a layer. At 16 lanes × (say) 64-wide tiles, the compute easily outruns the DDR3 feed — confirming the design is **bandwidth-bound, which is the point** (compute is cheap, bytes are the cost).

### 3.4 DDR3 weight-streaming schedule

- **Layout:** store each weight matrix **row-major, 5-ternary-per-byte**, aligned to DDR3 burst length (BL8 → 16 bytes/burst on x16). Stream a matrix as long sequential bursts to maximize MIG efficiency (sustained ≫ random).
- **Double-buffer** weight tiles in BRAM (ping-pong) so DDR3 reads overlap compute. With ~225 KB BRAM, reserve ~64–96 KB for the weight ping-pong, the rest for activations + KV window.
- **Schedule per decode step:** for each of the 24 layers, stream {Q,K,V,O} attn projections then {gate,up,down} FFN projections; interleave KV read/write. Total weight traffic per token = **model size in packed bytes ≈ 60 MB**.

### 3.5 Hand-RTL vs Vitis HLS split

| Block | Implementation | Why |
|---|---|---|
| Sign-select array + adder trees | **Hand-written Verilog** | Hot loop; need exact LUT/CARRY4 packing and timing closure at target Fmax |
| Weight unpacker / DDR3 stream FSM | **Hand RTL** around Xilinx **MIG** core | Burst alignment and double-buffering are timing-critical |
| RMSNorm / RoPE / softmax / quant | **Vitis HLS** | Math-heavy, low duty cycle, fast to iterate; HLS schedules DSP use |
| Top-level orchestration, layer loop | **HLS or small MicroBlaze/FSM** | Control, not throughput; UART/Ethernet glue |
| Verification | **verilator + cocotb** (already installed) | Bit-exact vs a Python BitNet reference + the 2B golden model |

Toolchain present on worker4: **Vivado 2025.2, verilator, cocotb, openFPGALoader** — sufficient end-to-end (simulate → synthesize → bitstream → flash).

---

## 4. Benchmark vs the RTX 3060 — exact methodology

**Same model, same tokenizer, same prompts, both devices.** Run the *identical* ternary model (the ~300M for a fair both-fit comparison, and optionally the 2B on GPU-only to show the GPU's scale advantage).

### 4.1 Metrics

1. **Energy per token (J/tok)** — the headline. = (mean board power during decode) ÷ (decode tokens/sec).
2. **Decode throughput (tok/s)** at **batch-1** (the deployment-relevant regime) and the GPU's best batch (to show its ceiling).
3. **Batch-1 latency (ms/token, TPOT)** and **time-to-first-token**.
4. **Tokens/sec/W** (efficiency), reported alongside raw tok/s so the trade is explicit.

### 4.2 Reference model + quantization (apples-to-apples)

- **Model:** one ternary BitNet b1.58 checkpoint (the on-board ~300M). Weights {−1,0,+1}, **8-bit activations (W1.58A8)** on *both* sides.
- **GPU stack:** `bitnet.cpp` / `llama.cpp` CUDA with the **GGUF `i2_s`** ternary kernels ([microsoft/BitNet](https://github.com/microsoft/BitNet)) so the GPU runs *the same ternary numerics* (its honest best case), plus a standard FP16/INT8 build to show what the GPU "naturally" does.
- **FPGA stack:** the RTL engine, bit-exact-validated against the same checkpoint via cocotb before any timing claim.
- **Decode-only, batch-1, fixed 256-token generation** from a fixed prompt set; report mean ± std over ≥5 runs; warm caches; exclude load time.

### 4.3 Power measurement (the part people get wrong)

- **GPU:** `nvidia-smi --query-gpu=power.draw --format=csv -lms 100` sampled across the decode window; integrate to energy. Report **board power** (the 170 W-class GA106 rail), and separately note this excludes the host CPU. Optionally a wall meter (Kill-A-Watt) on the whole machine minus idle, as a cross-check.
- **FPGA:** the Arty exposes power rails; measure with (a) an **inline USB/barrel power meter** on the board's 5 V input (whole-board watts — the honest number), and (b) **Vivado's power report** + on-board voltage monitor as a model cross-check. Report **whole-board watts**, not just FPGA core, so it's comparable to the GPU's board number.
- **Apples-to-apples rule:** compare *board-to-board* energy/token (GPU card vs FPGA board), state explicitly that neither includes its host, and never compare FPGA-core-only watts to GPU-board watts.

### 4.4 Cross-checks against literature

Validate the FPGA number sits in the right neighborhood: TeLLMe (KV260, 0.7B) = ~0.74 J/tok implied (7 W ÷ 9.51 tok/s); TerEffic (370M on-chip) = 1/455 ≈ **2.2 mJ/tok** (but on-chip, larger FPGA). The Arty, DDR3-bound, should land between — see §5.

---

## 5. Honest expected numbers

**Assumptions:** ~300M ternary model, **60 MB packed weights/token**; DDR3 sustained **0.6 GB/s** (mid of 0.5–0.8 honest range; 1.3 GB/s peak derated for MIG/UI/refresh); board **~3 W**; activations/KV ~10–20% traffic overhead folded in. 3060 numbers from public llama.cpp batch-1 decode + 170 W TDP.

| Metric | Arty A7-35T (this design) | RTX 3060 12 GB | Who wins |
|---|---|---|---|
| Weight traffic / token | ~60 MB (1.58 b) | ~60 MB (same ternary GGUF) or ~600 MB (FP16) | tie (ternary) |
| **Decode throughput, batch-1** | **~8–12 tok/s** (DDR3-bound: 0.6 GB/s ÷ 60 MB ≈ 10) | **~80–150 tok/s** (300M is tiny for 360 GB/s) | **GPU, ~10–15×** |
| Board power (decode) | **~3 W** | **~110–170 W** (rarely saturates on tiny batch-1) | **FPGA, ~40–55×** |
| **Energy / token** | **~0.25–0.4 J/tok** | **~1–2 J/tok** (e.g., 130 W ÷ 100 tok/s = 1.3 J) | **FPGA, ~4–8×** |
| **Tokens/sec/W** | **~3–4 tok/s/W** | **~0.6–0.9 tok/s/W** | **FPGA, ~4–6×** |
| Batch-1 latency (TPOT) | ~85–125 ms | ~7–13 ms | GPU |
| Max model that fits | ~300–400M (256 MB DDR3) | up to ~14B (12 GB) | GPU |

**Honest read of the trade.** The 3060 **wins raw tokens/sec by ~10–15×** and wins absolute latency and model scale outright. The Arty **wins energy/token and perf/watt** — but note the *realistic* margin here is **~4–8×, not 20–40×**, *because the 3060 idles inefficiently on a tiny batch-1 ternary model* (it can't fill 360 GB/s or 170 W usefully). The ~20–40× headline is achievable only against a **less efficient baseline** (e.g., the Jetson Orin Nano comparison TerEffic used = 19×, or a server GPU forced to dequantize FP16 at ~600 MB/token → ~10× more GPU traffic). **Stated plainly: the defensible, must-clear claim is FPGA ≈ 4–8× better energy/token than the 3060 on the same ternary model; the 20–40× figure requires a weaker baseline or the FP16-dequant GPU path, and should be reported as such — not as the headline number.** This honesty is the credibility of the whole program.

---

## 6. Milestones

**2-week (prove the multiply-free core):**
- Python BitNet b1.58 reference (load a ~300M or the 2B as oracle); export a GEMV test layer.
- Hand-RTL **sign-select + adder-tree GEMV lane**; verilator + cocotb **bit-exact** vs Python on one layer.
- Synthesize in Vivado 2025.2; confirm **0 DSP in the multiply path**, get LUT/Fmax for one lane.

**6-week (a layer streaming from DDR3):**
- Integrate **MIG DDR3** controller; weight unpacker (5-ternary/byte); double-buffered tile streaming.
- Run **one full transformer block** (attn + FFN, RMSNorm/RoPE/squared-ReLU) on-board; measure actual sustained DDR3 GB/s and per-layer cycles.
- First **measured board power** (inline meter) on a streaming workload.

**12-week (end-to-end decode + benchmark):**
- Full ~300M model decode loop, UART/Ethernet token I/O, KV in DDR3.
- Run the **§4 benchmark**: energy/token + tok/s on Arty vs the same ternary model on the 3060 via bitnet.cpp/llama.cpp, both power-instrumented.
- Write up the measured energy/token ratio with the honest baseline caveats from §5.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **"This won't beat the GPU, so why bother?"** (the big one) | Correct framing: the goal is **never** tokens/sec — it is **energy/token and batch-1 capability**. The GPU *will* win throughput; we win perf/watt (~4–8× measured) and demonstrate native ternary in silicon the GPU lacks. Report both columns honestly; the win is the ratio, not the rate. |
| DDR3 sustained bandwidth below 0.6 GB/s | Sequential burst layout + double-buffering + BL8 alignment maximize MIG efficiency; if still low, shrink the model (150–200M) so bytes/token drop proportionally — token rate is byte-bound, so a smaller model directly recovers tok/s. |
| 225 KB BRAM too small for KV + activations + weight tiles | Keep KV and embeddings in DDR3, stream per-layer; use INT8 KV; cap demo context at 512–1024; only a working window lives in BRAM. |
| 20,800 LUTs can't fit enough lanes | Compute is *not* the bottleneck (design is bandwidth-bound) — even 4–8 lanes outrun 0.6 GB/s. Right-size lanes to the DDR3 feed, not to peak FLOPs. |
| Embedding/LM-head (vocab 128,256) dominates memory | Use 8-bit embeddings, or a small-vocab distilled model; tie input/output embeddings; keep them in DDR3 (touched once per token, not per layer). |
| 2B doesn't fit (1.19 GB) — scope creep toward it | Hard rule: 2B is **GPU-baseline + RTL oracle only**; on-board model stays ≤~350M. Documented in §3.2. |
| GPU baseline "cheats" by being inefficient at batch-1 | Report *multiple* GPU baselines: (a) same ternary GGUF (its best case, ~4–8× gap), (b) FP16-dequant path (~10×+ gap), (c) cite TerEffic's Jetson/A100 comparisons (8–19×) for context — never a single cherry-picked number. |
| Timing closure / DDR3 bring-up eats the schedule | DDR3 MIG bring-up is the classic Arty time-sink; the 6-week milestone is *explicitly* the DDR3 integration gate. Use Digilent's reference MIG project as the starting point. |

---

## 8. Key resources (verified URLs)

**Ternary models & frameworks**
- BitNet b1.58 (original): https://arxiv.org/abs/2402.17764
- BitNet b1.58-2B-4T technical report: https://arxiv.org/abs/2504.12285
- BitNet 2B model + config (30L/2560/6912/20H/5KV, vocab 128256): https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- BitNet 2B GGUF (`i2_s`, 1.19 GB): https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
- bitnet.cpp (official 1-bit inference, energy/speedup numbers): https://github.com/microsoft/BitNet
- T-MAC (LUT mpGEMM, paper): https://arxiv.org/abs/2407.00088
- T-MAC (code): https://github.com/microsoft/T-MAC

**Ternary / low-bit FPGA accelerators (the SOTA to benchmark against)**
- TerEffic (370M on-chip 455 tok/s/W; 2.7B HBM 16 tok/s/W): https://arxiv.org/abs/2502.16473
- TeLLMe (KV260, 0.7B, 9.51 tok/s <7 W, table-lookup ternary matmul): https://arxiv.org/abs/2504.16266
- TeLLMe v2 (25 tok/s decode, 5 W): https://arxiv.org/abs/2510.15926
- TENET (sparsity-aware LUT-centric ternary, edge): https://arxiv.org/html/2509.13765v1
- FlightLLM (Alveo U280, configurable sparse DSP chain, 6× energy eff. vs V100S): https://arxiv.org/abs/2401.03868
- LUT Tensor Core (HW/SW co-design for LUT low-bit GEMM): https://arxiv.org/pdf/2408.06003

**The decode-wall / batch-1 framing**
- "Memory-Bound but Not Bandwidth-Limited" (batch-1 decode wall): https://arxiv.org/abs/2605.30571

**Hardware**
- Arty A7 reference manual (256 MB DDR3L, MT41K128M16, x16): https://digilent.com/reference/programmable-logic/arty-a7/reference-manual
- Micron MT41K128M16 DDR3L (2Gb, 128M×16): https://www.micron.com/products/dram/ddr3-sdram/part-catalog/mt41k128m16jt-125
- RTX 3060 12 GB (GA106, 360 GB/s, 170 W TDP) — NVIDIA: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/
