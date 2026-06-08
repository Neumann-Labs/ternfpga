# Direction B — The GPU's Low-Power Wingman: Heterogeneous Speculative Decoding with an Arty A7-35T Draft Engine

**One-line thesis.** A $130 Artix-7 FPGA can host a tiny *ternary* draft model that proposes K tokens for the RTX 3060's larger target model to verify in one batched pass — but the 100 Mb-Ethernet/UART round-trip is the make-or-break variable, and a brutally honest latency-budget analysis shows the *naive* per-step FPGA→GPU draft loop is killed by the link; the engineering win comes from (a) **batched/pipelined transport** that amortizes the link over many tokens, and failing that, (b) a pivot to the FPGA as an **always-on ultra-low-power router/gate** that decides *when* the GPU needs to run at all.

---

## 1. The problem & why the ML-Sys industry cares

Autoregressive LLM decoding is **sequential and memory-bandwidth-bound at batch 1**. Each new token requires streaming the *entire* weight matrix through the compute units to produce a single matrix-vector product, so each weight byte is reused exactly once. Arithmetic intensity collapses far below the hardware's compute/bandwidth balance point, leaving the ALUs idle while the memory system is saturated ([APXML, LLM inference bottlenecks](https://apxml.com/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference)). This is the central pain point of single-stream/edge/agentic inference: you cannot make one user's token stream faster just by adding FLOPs.

A 2026 measurement paper, *"Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode"* (Chen, [arXiv:2605.30571](https://arxiv.org/abs/2605.30571)), quantifies how bad it is and adds a twist: on an L4 GPU, Qwen-2.5-7B batch-1 decode hits ~81% of its analytic memory floor (≈62 ms/step baseline; 17.4 ms/step with GPTQ+ExLlamaV2), but on an H100 the *same model reaches only 27%* of peak bandwidth because **kernel-launch overhead dominates** once the GPU is fast enough (CUDA Graphs buys 1.259× on H100 vs only 1.028× on L4). Translation: throwing a bigger GPU at batch-1 latency has sharply diminishing returns. The structural fix the field converged on is **speculative decoding**.

**Speculative decoding** (Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192); concurrently Chen et al.) breaks the sequential dependency *without changing the output distribution*. A cheap **draft** model autoregressively guesses γ tokens; the expensive **target** model verifies all γ in a *single parallel forward pass* (the same cost as decoding one token, because the verify pass is also memory-bound and γ extra tokens ride along nearly for free). A modified rejection-sampling rule keeps the result *mathematically identical* to sampling from the target alone. Reported gains: **2×–3×** on T5-XXL with identical outputs.

The lineage the industry actually deploys:
- **Vanilla spec-decode** — separate small draft model (Leviathan, [2211.17192](https://arxiv.org/abs/2211.17192)).
- **Medusa** (Cai et al., [arXiv:2401.10774](https://arxiv.org/abs/2401.10774), [code](https://github.com/FasterDecoding/Medusa)) — no separate draft model; bolt extra "Medusa heads" onto the target to predict tokens t+2, t+3, … and verify a *tree* of candidates with tree-attention. **2.2× (Medusa-1)** to **2.3–3.6× (Medusa-2)**.
- **EAGLE / EAGLE-2 / EAGLE-3** (Li et al., [arXiv:2401.15077](https://arxiv.org/abs/2401.15077), [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)) — autoregress at the *feature* level (the hidden state before the LM head) instead of the token level, which removes most of the draft model's uncertainty. EAGLE-2 uses a *context-dependent dynamic draft tree* (draft confidence ≈ acceptance probability). Reported **acceptance length τ = 4.7–4.98 tokens/cycle** and **3.05×–4.26× speedup** on Vicuna/LLaMA2-Chat 7B–13B.

Why the FPGA angle is interesting: **every one of these methods spends GPU resources (VRAM, SMs, power) running the draft.** Medusa heads live in GPU VRAM; EAGLE's draft transformer runs on the same GPU and contends for the same memory bus. If the draft could run on a *separate, ~2 W ternary ASIC-like fabric*, the GPU would be freed to do nothing but verify — and the *system* energy/token could drop even if raw tokens/sec doesn't. That is the thesis of Direction B.

---

## 2. The core idea & the FPGA edge (precisely what the GPU structurally CANNOT do)

**The system.** worker4 hosts both an RTX 3060 12 GB (~360 GB/s, ~170 W) and an Arty A7-35T (~$130, 90 DSP48, 20,800 LUTs, 1.8 Mb BRAM, 256 MB DDR3L, 100 Mb Ethernet + USB-UART, **no PCIe to host**). The GPU runs the **target** (e.g. Mistral-7B Q4). The FPGA runs a **tiny ternary draft model** — the *same datapath built in Direction A* — proposing K tokens that the GPU verifies.

**What the GPU structurally cannot do, and the FPGA can:**

1. **Native ternary {−1, 0, +1} matmul with zero multipliers.** A ternary weight turns every multiply into a *select/add/subtract*. On the FPGA this is LUT logic at ~1 op/LUT/cycle with no DSP pressure; the GPU must still pay for INT8/INT4 tensor-core lanes that were never designed for 1.58-bit and waste silicon and energy on a datapath wider than the data. BitNet b1.58 (Ma et al., [arXiv:2402.17764](https://arxiv.org/abs/2402.17764); [Microsoft BitNet](https://github.com/microsoft/BitNet)) established that ternary LLMs are accuracy-competitive, making a ternary draft *viable*, not a toy.

2. **Always-on at single-digit watts.** The 3060 idles at tens of watts and spins up to ~170 W under load; an Artix-7 design of this size sits at **~1.5–3 W total board power**. The FPGA can run continuously as a wingman; the GPU cannot be left hot for free.

3. **Deterministic, kernel-launch-free latency.** The FPGA pipeline has *no CUDA kernel-launch tax* — the exact overhead [2605.30571](https://arxiv.org/abs/2605.30571) shows dominates fast-GPU batch-1 decode. A draft token emerges every fixed N cycles.

**What the FPGA structurally cannot do (the honest other side):** it has **~280× less memory bandwidth** than the 3060 and **~1.8 Mb** of fast on-chip RAM. It will *lose raw tokens/sec*. The credible wins are **perf-per-watt**, **freeing the GPU**, and **capability** (native ternary) — *never* raw throughput. And — the crux of this dossier — the FPGA is separated from the GPU by a **100 Mb link, not PCIe**. Whether draft tokens arrive fast enough is Section 4's central question.

---

## 3. Technical design on the Arty A7-35T

### 3.1 Reference target: the resource reality vs. published ternary FPGA work

The strongest published ternary-FPGA result is **TerEffic** (Chen et al., [arXiv:2502.16473](https://arxiv.org/abs/2502.16473)). It is the right *technique* reference and the wrong *scale* reference, and being precise about the gap is essential:

| | **TerEffic platform** | **Arty A7-35T (ours)** | Ratio |
|---|---|---|---|
| Part | AMD Alveo U280 (datacenter) | Xilinx XC7A35T (hobby) | — |
| On-chip SRAM | 8.85 MB BRAM + 33.75 MB URAM = **~42 MB** | **1.8 Mb = 0.225 MB** BRAM, no URAM | **~187×** |
| DSP slices | 3,041 | **90** | **~34×** |
| LUTs | ~780 K | **20,800** | **~37×** |
| Off-chip | 8 GB HBM @ 460 GB/s | 256 MB DDR3L @ ~2.67 GB/s peak | **~172×** BW |

TerEffic's headline 370M-param model running **16,300 tok/s @ 35.8 W (455 tok/s/W)** uses **two U280s** and 58 MB of on-chip storage. **None of that fits on an Arty.** Its 7B-class number (~290 tok/s @ 46 W, projected) is HBM-resident. We must scale *down by ~35×* and design for a model that actually fits.

### 3.2 What model actually fits as the draft

Two tiers, depending on whether weights live on-chip or stream from DDR3:

**Tier 1 — fully on-chip (the latency-optimal draft).** 1.8 Mb BRAM ≈ 225 KB. At 1.6 bits/ternary-weight (TerEffic's packing) that is **~1.1M ternary weights** of *weights-resident* capacity, minus KV cache and activations — realistically a **~0.5–1.0M-param** transformer (e.g. d_model=256, 4 layers, vocab tied/small). This is *tiny* but its entire weight set is one cycle away; no DDR3 stalls. This is the only configuration where the FPGA draft is genuinely fast.

**Tier 2 — DDR3-streamed weights.** 256 MB DDR3L holds up to **~1.28B ternary weights** at 1.6 bit. But sustained bandwidth is ~0.5–0.8 GB/s realistically (brief's figure; 2.67 GB/s peak, 16-bit bus @ 667 MHz). At 0.8 GB/s a model that touches **W bytes/token** runs at ≤ 0.8e9 / W tokens/s. A 30M-param ternary model ≈ 6 MB/token → **~130 tok/s ceiling from bandwidth alone** — *before* compute. A 100M-param model ≈ 20 MB → **~40 tok/s**. **DDR3 bandwidth, not compute, is the wall**, exactly as it is on the GPU but ~280× lower.

> **Design decision:** the draft is a **Tier-1, fully-on-chip ~1M-param ternary model.** It is the only choice that can plausibly out-pace the GPU's per-token clock (Section 4). A larger DDR3-streamed draft is *slower per token than the GPU itself*, which defeats the purpose.

### 3.3 The datapath (reuse Direction A's ternary core)

```
                 ┌──────────────────────── Arty A7-35T ───────────────────────┐
 host (worker4)  │                                                            │
   100 Mb Eth ───┼─► [MAC/UDP RX] ─► [token+state FIFO] ─► [Ternary Decoder]  │
   (LiteEth)     │                                            │  core (DirA)  │
                 │                                            ▼               │
                 │   embed(BRAM) ─► {QKV proj, attn, FFN} all ternary GEMV    │
                 │      via LUT-based select-add trees (NO DSP for matmul)     │
                 │      90 DSP reserved for LayerNorm/softmax scaling/RoPE     │
                 │   KV cache: BRAM ring buffer (few-hundred-token window)     │
                 │                                            │               │
                 │            [argmax/top-k sampler] ◄────────┘               │
                 │                     │                                      │
   100 Mb Eth ◄──┼── [UDP TX] ◄─ [K-token + draft-logprob packer]            │
                 └────────────────────────────────────────────────────────────┘
```

- **Ternary GEMV engine:** weights packed 2 bits each in BRAM; each MAC is `acc += (w==+1)? a : (w==-1)? -a : 0`. A LUT6 implements a 2:1 ternary select; an adder tree of depth log2(d_model) reduces. With d_model=256 and ~10k LUTs budgeted for the MAC fabric we sustain **on the order of a few thousand ternary-MACs/cycle @ 100 MHz**. One token through a ~1M-param model ≈ ~2M ternary-MACs → **~hundreds of µs/token of pure compute** when weights are on-chip (no DDR3 stall).
- **DSP usage:** the 90 DSP48s are *not* used for the ternary matmul (that's the whole point). They serve LayerNorm reciprocal-sqrt, softmax exp-approximation, RoPE sin/cos, and the per-channel INT scaling — the parts that aren't ternary.
- **Sampler:** argmax for greedy draft, or a small top-k for sampled draft; ship both the token IDs *and* the draft logprob q(x) per token so the GPU can run exact rejection sampling.

### 3.4 Hand-RTL vs. Vitis HLS split

| Block | Implementation | Why |
|---|---|---|
| Ternary GEMV adder-tree, BRAM weight packing | **Hand-written Verilog/SystemVerilog** | The whole perf-per-watt thesis lives here; HLS will not pack LUTs as tightly. Verify with verilator + cocotb. |
| Attention/KV-cache control FSM, RoPE | **Vitis HLS** | Control-heavy, iterate fast, modest area. |
| LayerNorm / softmax (DSP) | **Vitis HLS** | Fixed-point math pragmas are exactly HLS's strength. |
| Ethernet MAC + UDP | **LiteEth IP** (open-source) or Xilinx AXI-Ethernet-Lite | Don't hand-roll a MAC. |
| UART (debug/fallback) | Stock Xilinx UARTLite | Bring-up + the pivot path. |

---

## 4. The latency budget — the make-or-break analysis

This is the section that decides whether Direction B is real. The question: **for the FPGA-as-draft loop to beat GPU-only decode, K draft tokens must be produced *and shipped to the GPU* in less time than the GPU would have spent decoding those K tokens itself.**

### 4.1 The numbers we're working with (all from verified sources)

- **GPU target verify pass (Mistral-7B Q4, batch 1, RTX 3060):** the 3060 does ~40–45 tok/s on 7B Q4 ([singhajit benchmarks](https://singhajit.com/llm-inference-speed-comparison/), [mustafa.net benchmarks](https://mustafa.net/llm-tokens-per-second-benchmarks/)) → **~22–25 ms per decode step**, and a verify pass over γ tokens costs *essentially the same* (one memory-bound forward) → call it **~25 ms/verify**.
- **GPU-only baseline:** **~25 ms/token.**
- **FPGA draft compute (Tier-1, on-chip ~1M param):** hundreds of µs/token, say **~0.3–0.7 ms/token** → **~2–4 ms for K=6 tokens.**
- **Link — 100 Mb Ethernet:** an EAGLE-style payload is tiny (K token IDs + K logprobs ≈ a few hundred bytes). Wire time at 100 Mb/s is **<0.1 ms**; realistic UDP round-trip on a LAN with the FPGA's MAC and the host network stack is **~0.2–1 ms** (consistent with the single-digit-ms LAN figures in [Intel FPGA round-trip latency docs](https://www.intel.com/content/www/us/en/docs/programmable/848477/25-1/round-trip-latency.html) and [Electric UI's latency comparison](https://electricui.com/blog/latency-comparison)). **The kicker is the host-side per-packet/interrupt and the OS scheduler jitter, which can spike to several ms.**
- **Link — USB-UART (the trap):** at 57600 baud, 12 bytes ≈ **2 ms one way** ([Electric UI](https://electricui.com/blog/latency-comparison)); even at 3 Mbaud (FT2232H max) a few hundred bytes is ~1 ms each way plus USB micro-frame latency (~1 ms). **UART is too slow and too jittery to be the per-step draft transport.** Use Ethernet; keep UART for bring-up only.

### 4.2 The decisive comparison

**Naive synchronous loop** (FPGA drafts K, ships them, GPU verifies, GPU ships the accepted prefix + new state back, repeat):

```
T_cycle = T_draft(K) + T_link_up + T_verify + T_link_down
        ≈  3 ms      +  ~0.5 ms  +  25 ms   +  ~0.5 ms      ≈ 29 ms   per cycle
Accepted tokens/cycle (EAGLE-like α): τ ≈ 3–4 with a *weak* 1M draft (not the full ~4.7; our draft is tiny)
⇒ effective ~29 ms / 3.5 tokens ≈ 8.3 ms/token   vs GPU-only 25 ms/token  ⇒ ~3× — IF the link is well-behaved.
```

**The honest catch — link jitter eats the margin.** The win above assumes the round-trip *adds only ~1 ms total*. But the *time the GPU saves* per cycle is `(τ−1) × 25 ms ≈ 62 ms` of would-be GPU decode, against a link cost of ~1 ms — so a *clean* link wins comfortably. **The danger is not mean latency; it is tail latency.** If the host's network stack / scheduler injects a 5–10 ms stall *on every cycle*, and a tiny draft yields only τ≈2–3, the math tightens fast. With τ=2 and a 10 ms link round-trip: `(3 + 10 + 25 + ... )/2 ≈ 24 ms/token` — *break-even with GPU-only*. **So Direction B lives or dies on (a) draft acceptance rate and (b) link tail latency, not on FPGA compute.**

### 4.3 The fix that makes it real: pipelined / asynchronous drafting

Do **not** run synchronously. Decouple the link from the critical path:

1. **Speculative streaming with continuation.** The FPGA doesn't wait for the GPU's verdict to start the next draft. It keeps drafting *forward* along its own best guess and ships a *rolling window* of K-token packets. The GPU verifies and only occasionally sends a "rollback to token j, here's the corrected state" message. This **hides the ~25 ms verify and the link entirely behind continuous drafting** — the link is no longer in the per-token critical path; it only needs to keep up with *bandwidth* (trivial: a few KB/s), not meet a per-token *latency* deadline.
2. **Batch the transport.** Ship K=8–16 tokens per UDP datagram so one packet = one verify-worth of work; per-packet host overhead is amortized over K tokens.
3. **Result.** Steady-state throughput becomes `max(T_verify/τ, T_draft_per_token, T_link_per_token)`. With τ≈3, that's `25/3 ≈ 8.3 ms/token` GPU-bound — **the GPU is the bottleneck, the FPGA and link are hidden** — for a **~3× speedup** while the *draft work is off the GPU at ~2 W*.

### 4.4 If the link still kills it — the pivot (and it's a good one)

If, on real hardware, the 100 Mb-Eth + host-stack tail latency proves un-tamable (NIC interrupt coalescing, OS scheduler, no kernel-bypass), the *per-step* spec-decode collapses to break-even and is not worth the complexity. **Pivot to the FPGA as an always-on ultra-low-power GATE/ROUTER, not a per-token drafter:**

- The Arty runs a tiny ternary classifier/LM continuously at ~2 W and **decides whether the 3060 needs to wake up at all** — wake-word detection, intent/route classification, "is this query trivial enough to answer from the tiny model?", or **draft-a-whole-sentence then have the GPU verify once per sentence** (amortizing the link over 20–40 tokens, where even a 10 ms round-trip is <0.5 ms/token).
- This *embraces* the link latency (one round-trip per sentence, not per token) and *keeps the genuine wins*: the GPU stays asleep (huge idle-energy savings vs. a hot 170 W card), and the FPGA's native-ternary, deterministic, single-watt operation is exactly what a GPU cannot match. **The capability/energy story survives even if the latency story for fine-grained spec-decode does not.**

---

## 5. Benchmark vs. the RTX 3060 — exact methodology

**Goal:** apples-to-apples *system* comparison of three configurations producing **identical output distributions** (spec-decode is lossless by construction, so output quality is held constant — only speed and energy vary).

**Configurations:**
- **(i) GPU-only:** Mistral-7B Q4 on the 3060, llama.cpp / vLLM, batch 1, greedy + sampled.
- **(ii) CPU-draft spec-decode:** same target, draft = a small model on worker4's CPU (e.g. TinyLlama / a 1B Q4), spec-decode in llama.cpp/`transformers`. This is the *real* competitor — it has no link latency at all.
- **(iii) FPGA-draft spec-decode (this work):** same target, draft = the on-chip ternary ~1M model on the Arty, transport over 100 Mb Eth.

**Metrics (the only ones that matter):**
1. **Energy per token (J/tok)** — the headline. = (mean system power × wall-time) / tokens generated.
2. **Batch-1 latency (ms/token)** — time-to-token, steady state and p99 tail.
3. **Tokens/sec** — reported but *expected to lose to the GPU*; included for honesty.
4. **Acceptance length τ** — accepted tokens per verify (diagnoses draft quality).
5. **GPU duty cycle** — fraction of wall-time the 3060 is actually computing (the FPGA's value is *lowering this*).

**Power measurement — both sides, simultaneously:**
- **GPU:** `nvidia-smi --query-gpu=power.draw --format=csv -lms 100` (board-level, ~±5%) cross-checked against the PCIe-slot + 8-pin via a wall meter or a clamp if precision is needed. Subtract idle to get *marginal* decode energy, and also report *total* (idle matters for the always-on pivot).
- **FPGA:** the Arty has no onboard power telemetry → measure at the **barrel jack / USB with an inline USB power meter or a bench supply with current readout**; report total board watts. Vivado's post-implementation power report (XPE) gives a sanity-check estimate; the wall measurement is ground truth.
- **CPU draft (config ii):** RAPL via `powercap`/`turbostat` for package energy.
- **Wall-clock authority:** drive all three from one harness on worker4 that timestamps token emission; the *system* boundary includes GPU + (FPGA or CPU) so config (iii)'s FPGA watts are counted.

**Reference model + quantization held fixed:** Mistral-7B, Q4_K_M, identical prompts (MT-Bench subset + a code + a chat + a long-context prompt, mirroring EAGLE/Medusa eval sets so τ is comparable), temperature ∈ {0, 0.7}, 256-token generations, 20 runs, report median + p99.

**The apples-to-apples rule:** because spec-decode is distribution-preserving, all three emit the *same* tokens for a given seed — so we are *purely* measuring speed/energy of producing identical text. No quality caveat needed.

---

## 6. Honest expected numbers

Assumptions: 3060 batch-1 7B-Q4 ≈ 25 ms/tok, 170 W under load / ~15 W idle-ish floor; Arty draft ~2 W, ~0.5 ms/tok on-chip, τ≈3 (tiny draft → modest acceptance, *below* EAGLE-2's 4.7 because our draft is ~1000× smaller than a 7B-matched draft); link hidden by async pipelining (Section 4.3). **These are projections to be confirmed on hardware, not measured results.**

| Metric | (i) GPU-only | (ii) CPU-draft spec-dec | (iii) **FPGA-draft (ours)** | Verdict for (iii) |
|---|---|---|---|---|
| Tokens/sec (batch 1) | ~40 | ~90–110 (τ≈3, c small) | **~110** (GPU-bound, link hidden) | **WIN vs (i), ~tie (ii)** |
| Latency ms/tok (median) | ~25 | ~9–11 | **~9** | **WIN ~2.7× vs (i)** |
| Latency p99 (tail) | ~27 | ~12 | **~15–25** (link jitter risk) | **at risk — link tail** |
| GPU power under load (W) | ~170 | ~170 (draft also on GPU? no—CPU) ~170 | **~170 while verifying, but lower *duty*** | neutral peak |
| Draft power (W) | — (none) | ~30–60 (CPU package) | **~2 (FPGA)** | **WIN vs (ii)** |
| **Energy/token (J/tok)** | 170×0.025 ≈ **4.3** | (170+45)×0.010 ≈ **2.2** | (170+2)×0.009 ≈ **1.55** | **WIN: ~2.8× vs (i), ~1.4× vs (ii)** |
| GPU duty cycle | 100% | ~45% | **~45%** + GPU sleepable in pivot | **WIN (frees GPU)** |
| Output quality | reference | identical | **identical** (lossless) | tie by construction |
| Raw throughput vs 3060 | — | — | **LOSES** if FPGA ran solo | honest loss |

**Bottom line, stated plainly:**
- **Energy/token is the real, defensible win:** projected **~1.5 J/tok vs ~4.3 J/tok GPU-only (~2.8×)**, and notably **better than CPU-draft (~1.4×)** because the FPGA drafts at ~2 W where the CPU burns ~30–60 W.
- **The throughput "speedup" (~2.7×) is *borrowed* from spec-decode in general, not unique to the FPGA** — config (ii) gets most of it too. The FPGA's *unique* contribution is doing the draft at near-zero power and freeing the GPU, i.e. **the energy and duty-cycle columns, not the tokens/sec column.**
- **The single biggest risk to these numbers is the p99 tail** from the 100 Mb link + host stack. If async pipelining (4.3) can't hide it, (iii) degrades toward GPU-only and the **pivot (4.4)** becomes the actual deliverable — and that pivot *still* wins on idle energy.

---

## 7. Milestones

**2-week (prove the loop, in simulation + on the wire):**
- Verilator+cocotb model of the ternary GEMV core (reuse Direction A); validate one token of a ~1M-param ternary toy model bit-exact vs a NumPy reference.
- Stand up **LiteEth UDP** echo on the Arty; **measure real round-trip latency and p99 jitter** worker4↔Arty with a representative ~256-byte payload. *This single measurement decides whether 4.3 or 4.4 is the path.*
- Host harness: GPU-only Mistral-7B Q4 baseline (ms/tok, J/tok via nvidia-smi).

**6-week (end-to-end async spec-decode):**
- Synthesize the Tier-1 on-chip ternary draft on real Arty hardware; close timing @ 100 MHz; report LUT/DSP/BRAM utilization and **wall-measured board watts**.
- Implement the **asynchronous/pipelined** draft-stream + GPU-side rejection-sampling verifier (extend llama.cpp's spec-decode or `transformers`); achieve a *correct, lossless* end-to-end run.
- First real **(i) vs (iii)** energy/token + latency table; measure τ on MT-Bench prompts.

**12-week (the full three-way benchmark + the verdict):**
- Add **(ii) CPU-draft** competitor; produce the complete Section-5 table with p99 tails and GPU duty cycle, both *marginal* and *total* energy.
- Train/distill a *slightly* larger ternary draft (still on-chip or lightly DDR3-streamed) to push τ from ~3 toward ~4; quantify the τ-vs-area tradeoff.
- **Decide and document:** does async pipelining beat (ii) on energy at acceptable p99? If yes → fine-grained FPGA spec-decode is the result. If no → ship the **always-on gate/router pivot** with measured idle-energy savings as the result. Either way, a defensible energy-per-token number.

---

## 8. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **"This won't work because the 100 Mb-Eth round-trip is in the per-token critical path and a 5–10 ms tail erases the spec-decode win."** | **Highest** | **The answer:** don't put the link in the per-token path. Use **asynchronous continuation drafting** (4.3) so the link carries *bandwidth* (KB/s, trivial), not a per-token *deadline*; the verify + link hide behind continuous drafting. Measure p99 in week 2 before building more. If still fatal → **pivot to per-sentence verify / always-on gate** (4.4), which amortizes one round-trip over 20–40 tokens. |
| Tiny ~1M draft → low acceptance τ → small speedup | High | τ≈3 already yields ~2.7×; push toward 4 with distillation/EAGLE-style feature-level drafting. Honestly cap claims: a 1000×-smaller draft will not hit EAGLE-2's 4.7. |
| DDR3 bandwidth (~0.5–0.8 GB/s) caps any larger draft below the GPU's own per-token rate | High | **Keep the draft fully on-chip (Tier-1).** A DDR3-streamed draft is slower per token than the GPU → pointless. This bounds model size to ~1M params, accepted as a constraint. |
| Only 90 DSPs / 20,800 LUTs — can the ternary core even fit at 100 MHz? | Medium | Ternary matmul uses **LUTs, not DSPs**; DSPs reserved for norm/softmax/RoPE. TerEffic shows the technique scales; we're 35× smaller and 35× lower-target. Verilator area estimate in week 1, real util in week 6. |
| GPU verify pass cost grows with γ and KV length, eroding the "free verify" assumption | Medium | Keep γ=K modest (6–8); the verify is memory-bound so γ extra tokens are near-free until the tree blows up KV traffic — bound the draft tree like EAGLE-2's dynamic-but-capped tree. |
| Config (ii) CPU-draft is "good enough" and the FPGA adds complexity for marginal gain | Medium (to the *thesis*) | The FPGA's win is **energy/duty-cycle, not tokens/sec** — quantify the ~1.4× energy edge over CPU-draft and the GPU-sleep benefit; if that's not compelling, the pivot's always-on gate (continuous ~2 W classification the CPU can't match on energy) is. |
| No PCIe → can't share GPU VRAM / KV cache cheaply | Medium | Inherent. The async protocol ships only token IDs + logprobs + occasional rollback state (small); never tries to share KV over the wire. |
| Output-correctness bugs in the custom rejection-sampler break losslessness | Medium | Spec-decode is *provably* lossless — assert exact-match against GPU-only greedy decode (same seed) in CI; any divergence is a bug, caught immediately. |

**The one-sentence honest verdict:** the FPGA-as-per-token-drafter is *plausibly* a ~2.8× energy/token win over GPU-only and ~1.4× over CPU-draft **if and only if** asynchronous transport hides the 100 Mb link; if the link's tail latency proves un-hideable, the project still produces a genuinely valuable **always-on ~2 W ternary gate that lets the 170 W GPU sleep** — and that capability is something the GPU structurally cannot provide.

---

## 9. Key resources (verified URLs)

**Speculative decoding — algorithms:**
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023) — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) · [PMLR](https://proceedings.mlr.press/v202/leviathan23a.html)
- Cai et al., *Medusa: Multiple Decoding Heads* — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) · [code](https://github.com/FasterDecoding/Medusa)
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Li et al., *EAGLE-2: Dynamic Draft Trees* — [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- Xia et al., *Unlocking Efficiency in LLM Inference: A Survey of Speculative Decoding* — [arXiv:2401.07851](https://arxiv.org/html/2401.07851v3)

**Ternary models & ternary FPGA:**
- Ma et al., *BitNet b1.58 (The Era of 1-bit LLMs)* — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) · Microsoft BitNet [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)
- Chen et al., *TerEffic: Highly Efficient Ternary LLM Inference on FPGA* — [arXiv:2502.16473](https://arxiv.org/abs/2502.16473)
- *TeLLMe: Energy-Efficient Ternary LLM Accelerator for Edge FPGAs* — [arXiv:2504.16266](https://arxiv.org/abs/2504.16266)

**Batch-1 decode wall (the latency-headroom argument):**
- Chen, *Memory-Bound but Not Bandwidth-Limited: Batch-1 LLM Decode* — [arXiv:2605.30571](https://arxiv.org/abs/2605.30571)
- APXML, *Memory & Compute Bottlenecks in Inference* — [apxml.com](https://apxml.com/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference)

**Hardware / latency references:**
- Digilent *Arty A7 Reference Manual* — [digilent.com/reference/programmable-logic/arty-a7/reference-manual](https://digilent.com/reference/programmable-logic/arty-a7/reference-manual)
- Intel FPGA Ethernet *Round-Trip Latency* docs — [intel.com](https://www.intel.com/content/www/us/en/docs/programmable/848477/25-1/round-trip-latency.html)
- Electric UI, *Latency comparison across links (incl. UART)* — [electricui.com/blog/latency-comparison](https://electricui.com/blog/latency-comparison)
- RTX 3060 LLM benchmarks — [singhajit.com](https://singhajit.com/llm-inference-speed-comparison/) · [mustafa.net](https://mustafa.net/llm-tokens-per-second-benchmarks/)
