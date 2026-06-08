# Shared Build & Benchmark Methodology for a Ternary + Sparse LLM-Inference Engine (A+B+D)

**Thesis: One hand-built ternary+sparse compute core on a $130 Arty A7-35T — measured against an RTX 3060 in the same box (worker4) — credibly wins energy-per-token and batch-1 latency, exploits unstructured sparsity the GPU structurally cannot, and doubles as a speculative-decode draft engine; this document is the prod-safe, instrumented, apples-to-apples methodology that makes those three claims defensible with real silicon and real joules, never raw throughput.**

---

## 1. The problem & why the ML-Sys industry cares

LLM token generation (the "decode" phase, batch=1) is **memory-bandwidth bound, not compute bound**. To emit one token an autoregressive model must stream every weight from memory exactly once; arithmetic intensity is ~1-2 FLOP/byte, far below the ridge point of any modern accelerator's roofline. On a GPU the matrix-multiply units sit mostly idle during single-stream decode while HBM/GDDR bandwidth is the wall. This is now a first-class research concern — see XQuant, *"Breaking the Memory Wall for LLM Inference"* (arxiv.org/abs/2508.10395) and *"Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need"* (arxiv.org/html/2507.14397v1), both of which frame decode as a bandwidth/capacity problem rather than a FLOPs problem.

Two structural responses have emerged, and they compose:

1. **Extreme weight quantization.** BitNet b1.58 (Ma et al., arxiv.org/abs/2402.17764) trains LLMs whose every weight is **ternary {-1, 0, +1}** — log₂(3) ≈ **1.58 bits/weight** — and reports matching the perplexity and downstream accuracy of an FP16 LLaMA of the *same size and token budget*. Because the weights are ternary, the core operation becomes **multiplier-free**: a matmul degenerates into signed adds/subtracts and skips. Microsoft's `bitnet.cpp` (github.com/microsoft/BitNet) reports **2.37×-6.17× speedup and 71.9%-82.2% energy reduction on x86 CPUs** versus the FP16 baseline, and **55.4%-70.0% energy reduction on ARM** — by exploiting exactly this. The first open 2B-scale native-1-bit model and its measured properties are documented in the *BitNet b1.58 2B4T Technical Report* (arxiv.org/abs/2504.12285).

2. **Sparsity.** Real LLM activations and many weight matrices are highly sparse (ReLU-family and contextual activation sparsity routinely exceed 50-90% zeros). A GPU's SIMT lanes and dense tensor cores cannot cheaply skip *unstructured/dynamic* zeros — they pay for the multiply-by-zero. NVIDIA tensor cores accelerate only **2:4 structured** sparsity, a rigid pattern. This is the GPU's structural blind spot.

Why an ML-systems org (the kind that builds an in-house C training/inference stack mapped exactly to its hardware) cares: the win is **performance-per-watt and batch-1 latency determinism**, the two axes where a small, purpose-built datapath can beat a general-purpose GPU even though the GPU has ~280× more memory bandwidth. The published FPGA evidence is already strong: **TerEffic** (*Highly Efficient Ternary LLM Inference on FPGA*, arxiv.org/html/2502.16473) runs a 370M ternary model fully on-chip at **16,300 tok/s and 455 tokens/s/W**, reporting **19× better power-efficiency than a Jetson Orin Nano** and **8× better energy efficiency than an A100** for the HBM-assisted 2.7B variant — its ternary matmul core is built from **LUTs, not DSPs**. And **TeLLMe** (*An Energy-Efficient Ternary LLM Accelerator for Prefill and Decode on Edge FPGAs*, arxiv.org/html/2504.16266) runs a 0.7B ternary LLM on an **edge** FPGA at **<7 W**. These are the existence proofs this program stands on.

## 2. The core idea & the FPGA edge (what the GPU structurally CANNOT do)

**One reusable ternary+sparse MAC-free compute core** serves all three directions:

- **(A) Energy/token win.** Ternary weights mean the inner loop is *signed accumulate + skip*, never a multiply. On the FPGA this is LUT/adder logic, not DSP multipliers, so we pack far more "MAC-equivalent" ops per joule than a GPU spending FP16/INT8 multiplies. We compete on **tokens/joule and batch-1 latency**, the FPGA's home turf, and concede raw tokens/sec.
- **(D) Sparsity the GPU can't exploit.** The same core consumes a **compressed sparse weight/activation stream** (run-length / index-skip format) and *physically does not issue cycles for zeros*. There is no SIMT-lane divergence penalty and no fixed 2:4 pattern requirement — arbitrary, dynamic, per-token sparsity directly shrinks both DDR3 traffic and cycle count. This is the capability a GPU lacks: a GPU can store a sparse matrix but its dense MAC units still stride across the zeros.
- **(B) Speculative-decode draft engine.** In speculative decoding (survey: aclanthology.org/2024.findings-acl.456.pdf; *Unlocking Efficiency*, arxiv.org/pdf/2401.07851) a small fast **draft** model proposes k tokens that the large **target** model verifies in one parallel forward pass; accepted tokens are free. Acceptance rate, not draft throughput, governs the speedup. A's ternary datapath **is** B's draft engine: the FPGA proposes tokens at very low energy and deterministic latency while the 3060 runs the target/verifier. The draft model's job is cheap, low-latency proposals — exactly what a tiny bare-metal ternary core is best at.

**The structural GPU gaps, precisely:**
1. **No native ternary datapath** — a GPU must up-convert ternary to INT8/FP16 and multiply; it cannot natively skip the multiply. The FPGA's LUTs *are* the ternary ALU.
2. **No cheap unstructured/dynamic sparsity** — SIMT divergence + dense tensor cores; only rigid 2:4 is accelerated.
3. **Batch-1 latency floor + jitter** — kernel-launch overhead, dynamic scheduling, and the box-car power behavior all add nondeterminism; a statically-scheduled FPGA datapath is cycle-deterministic.

This is the same 8×8-class systolic/dataflow substrate the project's top-ranked directions already converge on (see `judging/TOP5.md` — #1 hand-authored cycle-exact systolic GEMM is the shared core); here it is specialized to **ternary weights + sparse skipping**, which is what makes A, B, and D literally the same RTL.

## 3. Technical design on the Arty A7-35T

**Device budget (XC7A35T, confirmed on worker4, IDCODE `0x0362d093`):** 20,800 LUTs, 41,600 FFs, **90 DSP48E1**, **1,800 Kb (225 KB) BRAM**, 256 MB DDR3L (16-bit @ 667 MT/s → **~1.33 GB/s theoretical, ~0.5-0.8 GB/s sustained**), 10/100 Ethernet, USB-UART.

**What fits (be precise).** A 7B model does **not** fit. At 1.58 bits/weight, on-chip BRAM (225 KB ≈ 1.84 Mb) holds ~**1.16M ternary weights** — enough for a *single small layer's* hot weights or the KV/activation working set, not a model. DDR3 (256 MB) at 1.58 bits/weight holds ~**1.3B ternary params** in principle, but the **sustained-bandwidth ceiling sets the real model size**: at ~0.6 GB/s sustained, streaming W weights once per token costs `W × 1.58/8 bytes ÷ 0.6 GB/s`. For an honest interactive target of a few tokens/sec, the realistic resident model is **~100M-300M ternary parameters** (e.g. ~100M params ⇒ ~20 MB ternary ⇒ ~30 ms/token weight-stream ⇒ ~30 tok/s bandwidth ceiling before compute/overhead; ~300M ⇒ ~60 MB ⇒ ~100 ms/token ⇒ ~10 tok/s). Sparsity (D) directly relaxes this: 70% weight sparsity cuts the streamed bytes ~3×. **Target reference model: a ~100M-class ternary LLM** (a BitNet-b1.58-style small model, distilled/quantized down — Section 4), with the published `bitnet_b1_58-large` (0.7B) used only as the *functional/accuracy oracle* on the GPU/CPU side, not as the on-Arty model.

**Datapath (the exact inner loop).** Weight-stationary 1-D/2-D array of **ternary processing elements (TPEs)**. Each TPE: takes an INT8 activation `a` and a 2-bit ternary weight code `w ∈ {-1,0,+1}`; computes `acc += (w==+1 ? a : w==-1 ? -a : 0)` — i.e. a sign-select + skip into a signed accumulator. **Zero DSPs on the multiply path** (this is the TerEffic insight: ternary matmul is LUT logic). The 90 DSP48E1 are reserved for the **non-ternary glue**: LayerNorm/RMSNorm, residual scales, the few full-precision projections, softmax reciprocals, and the activation requantization — work that genuinely needs multipliers. A ~16×16 TPE array (256 TPEs in LUTs) is feasible within 20,800 LUTs alongside control; the array size is tuned to keep the array fed at the DDR3 sustained rate (no point being wider than bandwidth allows — the design is deliberately bandwidth-matched).

**Sparsity engine (D).** Weights stream from DDR3 in a **bitmap/run-length sparse format**: a presence bitmap plus packed nonzero ternary codes. A skip-decoder advances the activation pointer and gates TPE enables so zero columns consume no MAC cycle and (with RLE) no DDR3 byte. This is the unit that the GPU has no analog for.

**Memory hierarchy.** DDR3-streamed: the per-layer ternary weight stream (the bulk). On-chip BRAM: double-buffered weight tiles (ping-pong to hide DDR3 latency), the activation vector, the KV cache for the small context, and partial sums. Classic dataflow tiling with weight/activation reuse is mandatory to approach the bandwidth roofline rather than the much-lower random-access rate.

**Hand-RTL vs Vitis HLS split.**
- **Hand-written SystemVerilog (the performance- and determinism-critical core):** the TPE, the systolic array + accumulator cascade (using DSP `PCIN/PCOUT` only where INT glue needs it), the double-buffer/tiling FSM, the sparse skip-decoder, and the DDR3↔BRAM streaming controller. These must be cycle-deterministic and area-tight — exactly the layer the C-stack ethos says you own by hand.
- **Vitis HLS (the surrounding, less timing-critical logic):** RMSNorm/softmax/requant blocks, the top-level token-loop sequencer, and the UART/Ethernet result-readback path. HLS buys productivity where cycle-exactness doesn't matter.
- **SoC scaffold:** a **LiteX** (github.com/enjoy-digital/litex) SoC providing LiteDRAM (DDR3 controller — known-good Arty bring-up), LiteEth (100 Mb), and a small VexRiscv/UART control plane to load weights, kick the token loop, and read back logits. This de-risks DDR3/Eth so effort concentrates on the core.

## 4. Benchmark vs the RTX 3060 — exact methodology

**Goal: apples-to-apples energy-per-token and batch-1 latency**, with raw tokens/sec reported honestly (and expected to lose). Both devices run on the same host, worker4 (Ryzen 9 5950X, Ubuntu 24.04).

### 4.1 GPU baseline setup on worker4 (PROD-SAFE — coordinated maintenance window)

worker4 currently uses the **open `nouveau`** driver and **runs the user's production Twenty CRM containers**. Installing the proprietary NVIDIA driver almost certainly requires **blacklisting nouveau + regenerating initramfs + a REBOOT** — because the blacklist must take effect in the initial ramdisk *before* nouveau loads (confirmed: nouveau and the proprietary module cannot co-reside, and the blacklist lives in initramfs — linuxconfig.org/how-to-disable-blacklist-nouveau-nvidia-driver-on-ubuntu-22-04-jammy-jellyfish-linux). **Therefore treat this as a coordinated maintenance window, not an inline step.**

Prod-safe sequence:
1. **Schedule a window.** Quiesce/announce Twenty CRM downtime; confirm `docker compose` services have `restart: unless-stopped`/`always` so they return on boot, and snapshot/back up the CRM DB volume first.
2. **Pin & pre-stage.** Note running kernel (`uname -r`); install the **server/headless** driver flavor to avoid pulling a desktop X stack onto a server: `nvidia-headless-<ver>` + `nvidia-utils-<ver>` (these exist precisely for compute-only/headless boxes — oneuptime.com/blog/post/2026-03-02-install-nvidia-drivers-ubuntu-server/view). Prefer Ubuntu's packaged driver (`ubuntu-drivers devices` → `apt install`) over the `.run` installer so DKMS + Secure Boot signing are handled.
3. **Blacklist nouveau + initramfs:** create `/etc/modprobe.d/blacklist-nouveau.conf` (`blacklist nouveau`, `options nouveau modeset=0`), `sudo update-initramfs -u`.
4. **Reboot in the window.** After boot: `lsmod | grep nouveau` must be empty; `nvidia-smi` must enumerate the 3060; **then verify the CRM containers came back healthy.**
5. **Persistence + clocks for stable energy numbers:** `sudo nvidia-smi -pm 1` (persistence mode), and record/optionally lock clocks (`nvidia-smi -lgc`) so the baseline is reproducible.

**No-reboot path — researched, offered as a best-effort, not promised.** It is *technically possible* to swap drivers live: stop everything using the GPU, `sudo modprobe -r nvidia_drm nvidia_modeset nvidia nvidia_uvm` (or, from nouveau, `modprobe -r nouveau`), then `sudo modprobe nvidia nvidia_uvm` (bbs.archlinux.org/viewtopic.php?id=269686; en.opensuse.org/SDB:NVIDIA_the_hard_way). **Caveat (be honest):** unloading nouveau requires nothing holds the framebuffer/DRM (no X, no console framebuffer in use), the blacklist still won't persist across the *next* reboot without the initramfs step, and a mismatched-but-loaded module can silently block CUDA. **Recommendation: do the clean reboot in a window.** Since the CRM is CPU/Postgres containers that don't use the GPU, the only real cost is the reboot itself — a no-reboot hack adds fragility for little gain. (Alternative if even one reboot is unacceptable: run the GPU baseline on a *different* CUDA box and treat worker4 purely as the FPGA host; the energy comparison is still valid because we measure the GPU's own power, not the host's.)

### 4.2 Reference model + quantization pipeline

- **Functional/accuracy oracle (shared ground truth):** **`microsoft/bitnet-b1.58-2B-4T`** (HF: huggingface.co/microsoft/bitnet-b1.58-2B-4T) with the GGUF build `microsoft/bitnet-b1.58-2B-4T-gguf` (`ggml-model-i2_s.gguf`). Download via `huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir models/...`.
- **CPU/GPU ternary baseline runner:** **`bitnet.cpp`** (github.com/microsoft/BitNet), which is built on llama.cpp (needs CMake ≥3.22, Clang ≥18, Python ≥3.9; `python setup_env.py`). Run: `python run_inference.py -m models/.../ggml-model-i2_s.gguf -p "..." -cnv`; benchmark: `python utils/e2e_benchmark.py -m <model> -n 200 -p 256 -t 4`. This is the **honest ternary CPU baseline** and the reference for "what ternary inference costs in software." For a **GPU** baseline, run the same/comparable model under **llama.cpp CUDA** (or vLLM for a dense FP16 control) so the 3060's tokens/sec and power are measured on the identical workload.
- **The on-Arty model:** a **~100M-class ternary model** — obtained by either (a) using a small public BitNet-style checkpoint (`1bitLLM/bitnet_b1_58-large` 0.7B as the upper bound, distilled/pruned down to ~100M to fit the bandwidth budget), or (b) quantization-aware training a ~100M Transformer to ternary with **Brevitas** (github.com/Xilinx/brevitas) and exporting weights to the packed sparse format the RTL consumes. Brevitas is the QAT front-end; the FPGA core is the back-end. (FINN — github.com/Xilinx/finn — is the dataflow alternative if we want a generated baseline to compare our hand-RTL against.)
- **Spec-decode pairing (B):** the on-Arty ~100M ternary model is the **draft**; the 3060 runs a larger compatible **target** (same tokenizer/vocab family — e.g. the 0.7B/2B BitNet as target, or a small Llama). Measure acceptance rate and end-to-end tokens/sec of the *coupled* system vs the target alone.

### 4.3 Energy measurement — both sides, apples-to-apples

**The unit is joules/token = (mean power during decode − idle power) integrated over the run ÷ tokens generated**, plus a separately-reported *wall* (total-power) number. Idle subtraction isolates the *marginal* cost of inference; total power is reported too so neither side hides static draw.

**GPU side.** Sample `nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu --format=csv -lms 100` (NVML under the hood) during a sustained decode run. **Critical accuracy caveat (cite and design around it):** nvidia-smi reports a **box-car average over a short window** and on recent GPUs only ~25% of runtime is actually sampled, giving up to ~70% error on short kernels (*Part-time Power Measurements: nvidia-smi's Lack of Attention*, arxiv.org/html/2312.02741v2). **Mitigation (their prescription):** run ≥**32 iterations / ≥5 s** of continuous decode, discard warm-up reps, run ≥4 trials with randomized inter-trial delays to phase-shift the sampling window; this cuts error from ~39% → ~5%. Use **NVML/DCGM at 10 Hz** and **integrate** P·Δt rather than eyeballing a peak. Per-token energy = ∫P dt / N_tokens with idle (persistence-mode, model loaded, no decode) subtracted. (Methodology cross-check: TokenPowerBench, arxiv.org/html/2512.03024v1.)

**FPGA side — two independent meters, cross-checked:**
1. **Inline USB/barrel power meter (primary, whole-board):** a cheap inline USB power meter (or a bench supply with current readout on the barrel jack) measures **total board watts** at the wall-of-the-board. Integrate over the run for joules; divide by tokens. This is the honest "what does the whole board cost" number.
2. **On-board XADC rail telemetry (fine-grained, FPGA-internal):** the Arty A7 has a built-in **5 V current-sense path** — a **5 mΩ resistor + TI INA199A1 amplifier producing 250 mV/A, fed to XADC Auxiliary Channel 9**, with the 5 V rail (÷5.99) on **AUX Channel 1** (Arty A7 Reference Manual, digilent.com/reference/programmable-logic/arty-a7/reference-manual). Configure both as unipolar, do a simultaneous conversion, and the FPGA **reports its own instantaneous V×I power** over UART per token — no external hardware needed. This gives a cycle-correlated power trace the GPU can't match.
3. **Vivado XPE / `report_power` (estimate, for sanity + breakdown):** the synthesized design's vectorless and SAIF-driven power report gives a static/dynamic and per-block estimate to validate the measured numbers and attribute power to compute vs DDR3.

Cross-check: USB-meter total ≈ XADC 5 V-rail total + 3.3 V-rail draw; Vivado estimate should land within tolerance. Report **measured**, not estimated, as the headline.

**Apples-to-apples rules:** identical prompt set, identical generated-token count, identical context length, batch=1 on both, decode-phase only for the per-token energy (prefill reported separately), warm caches/loaded weights on both, idle subtracted on both, ≥5 s continuous runs, ≥3 trials, report mean ± stdev. The model **need not be byte-identical** across devices (the Arty runs ~100M, the GPU may run a larger target) — so the **primary cross-device claim is tokens/joule and latency on each device's best-fit ternary model**, with a *same-model* point (run the ~100M ternary on both Arty and 3060 via llama.cpp/a tiny CUDA kernel) as the strict apples-to-apples anchor.

## 5. Honest expected numbers

Anchored to published edge-FPGA ternary results (TeLLMe: 0.7B ternary, <7 W, 9.51 tok/s decode on a Kria KV260 @250 MHz — a *bigger, faster, MPSoC* FPGA than the Arty) and TerEffic (370M on-chip, 455 tok/s/W on an Alveo U280 with HBM). The Arty A7-35T is **much smaller, has only ~0.6 GB/s sustained DDR3 (no HBM), and ~100 MHz core**, so it will land *below* those on tokens/sec but remain strong on tokens/joule. RTX 3060 figures are typical for batch-1 decode of a small model.

| Metric (batch-1 decode, ~100M-class ternary unless noted) | RTX 3060 (12 GB) | Arty A7-35T (this build) | Verdict |
|---|---|---|---|
| Raw tokens/sec | **~60-150** tok/s (bandwidth-rich) | **~5-20** tok/s (DDR3-bound) | **GPU wins ~5-15×** (expected) |
| Board/chip power during decode | ~80-120 W (of ~170 W TDP) | **~2-5 W** (whole board) | **FPGA wins ~25-40×** |
| **Energy per token (J/token)** | ~0.7-1.5 J/tok | **~0.2-0.6 J/tok** | **FPGA wins ~2-5×** ⭐ |
| Batch-1 latency jitter (token-to-token) | ms-scale, scheduler/launch jitter | **cycle-deterministic** (static schedule) | **FPGA wins on determinism** ⭐ |
| Unstructured/dynamic sparsity exploit | none (dense MACs; only 2:4) | **direct byte+cycle skip** (RLE sparse) | **FPGA-only capability** ⭐ |
| Native ternary datapath | no (up-converts to INT8/FP16) | **yes (LUT sign-select, 0 DSP mult)** | **FPGA-only capability** ⭐ |
| As spec-decode draft (coupled system) | — | draft @ low-J, deterministic latency | **enables GPU end-to-end speedup if accept-rate high** |

**Assumptions/caveats:** GPU tokens/sec and watts depend heavily on model size and kernel (llama.cpp CUDA vs vLLM); the 3060 number assumes a comparably small model so the comparison isn't a strawman against a 7B. The FPGA's J/token win **narrows or inverts** if DDR3 streaming dominates and the design can't hit the bandwidth roofline — hence the bandwidth-matched array sizing and sparsity. The energy win is the **load-bearing claim**; tokens/sec is conceded; sparsity + native ternary are **capability** wins independent of magnitude. nvidia-smi's measurement error (Section 4.3) is controlled by the long-run/multi-trial protocol — single-shot numbers are not reported.

## 6. Milestones

**Weeks 0-2 (de-risk the core + measurement, all in sim/CPU):**
- Stand up `bitnet.cpp` on worker4 CPU; reproduce a tokens/sec + (CPU RAPL) energy baseline on `bitnet-b1.58-2B-4T`. (No GPU needed yet — this de-risks the model/tooling before the maintenance window.)
- Hand-write the **ternary PE** (sign-select + skip) and an 8×8 TPE array in SystemVerilog; **verilator + cocotb** prove it bit-exact vs a NumPy ternary-matmul golden model (zero mismatch). gtkwave shows correct accumulate + zero-skip.
- Vivado synth of the array → first **utilization (LUT/DSP/BRAM) + Fmax** report; confirm DSP-free multiply path.

**Weeks 2-6 (real silicon, single-device numbers):**
- LiteX SoC on the Arty: **DDR3 (LiteDRAM) + UART + Eth** bring-up; stream a tiny ternary layer's weights from DDR3 through the array; read logits back over UART. `openFPGALoader` flashes the bitstream.
- Add the **sparse skip-decoder** (bitmap/RLE) and the **XADC self-power readback** (AUX ch 1 + 9). First **measured tokens/sec and J/token** for a single layer / tiny model on the Arty, cross-checked with the inline USB meter and `report_power`.
- **Coordinated maintenance window:** install the NVIDIA driver + CUDA on worker4 (Section 4.1); verify CRM healthy; stand up the llama.cpp-CUDA GPU baseline.

**Weeks 6-12 (the comparison + spec-decode):**
- Run the **full apples-to-apples** protocol: ~100M ternary model end-to-end on the Arty; same model on the 3060 (strict anchor) + the 0.7B/2B target on the 3060 (realistic). Produce the Section-5 table with **measured** mean±stdev J/token and latency, ≥3 trials, idle-subtracted, ≥5 s runs.
- **(B) Speculative decode:** wire the Arty as **draft** proposing k tokens, 3060 as **target/verifier**; measure **acceptance rate** and coupled end-to-end tokens/sec + total energy vs target-alone. Demonstrate the FPGA draft's energy/latency advantage.
- **(D)** Sweep activation/weight sparsity (e.g. 0/50/70/90%) and plot the **measured** byte-traffic and cycle/energy reduction the GPU cannot match.

## 7. Risks & mitigations

- **Biggest "this won't work because…": "the Arty is 280× short on memory bandwidth, so it just loses — there's no win."** **Answer:** correct *for raw throughput*, which we concede up front. The claim is **energy/token and batch-1 latency**, where published edge-FPGA ternary work (TeLLMe <7 W; TerEffic 455 tok/s/W, 8× an A100's energy efficiency) already demonstrates FPGAs winning the perf/watt axis. We size the array to the DDR3 roofline (not wider), use ternary to eliminate multiplies, and use sparsity to *cut the bytes streamed* — directly attacking the bandwidth wall. The energy win survives even at low tokens/sec.
- **DDR3 sustained bandwidth far below 1.33 GB/s for random access.** Mitigate with strict **sequential streaming + double-buffered tiling + on-chip reuse**; report the achieved fraction of roofline honestly (cf. the project's DMA-ledger direction, `judging/TOP5.md` #13).
- **nvidia-smi power error (~70% on short runs).** Mitigate with the ≥5 s / ≥32-iter / ≥4-trial phase-shifted protocol (arxiv.org/html/2312.02741v2) → ~5% error; integrate NVML at 10 Hz; report mean±stdev.
- **Driver install breaks prod (Twenty CRM).** Mitigate with the maintenance-window sequence, DB snapshot, headless driver flavor, `restart: always` verification, and a fallback of running the GPU baseline on a separate CUDA box (the energy comparison stays valid). No-reboot swap is offered but **not** relied upon.
- **~100M ternary model doesn't exist off-the-shelf at the right size.** Mitigate: distill/prune from `1bitLLM/bitnet_b1_58-large` (0.7B), or QAT a ~100M Transformer to ternary with Brevitas; the *accuracy oracle* stays the published 2B model so quality is anchored.
- **Spec-decode acceptance rate too low to net a win.** Mitigate: choose a draft trained on/distilled toward the target's distribution (acceptance is distribution-match-driven, per the surveys); even a modest accept rate still demonstrates the **draft-engine energy/latency** result, which is the FPGA contribution regardless of the coupled speedup.
- **Fitting array + control + sparse decoder in 20,800 LUTs.** Mitigate: ternary PEs are tiny (no DSP), start 8×8 and grow only to the bandwidth-matched size; offload norm/softmax glue to the 90 DSPs and HLS.

## 8. Key resources (URLs verified during research)

- BitNet b1.58 (ternary LLM, 1.58 bit): https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T Technical Report: https://arxiv.org/abs/2504.12285
- Microsoft BitNet / `bitnet.cpp` (inference framework, x86/ARM speed+energy): https://github.com/microsoft/BitNet
- BitNet-b1.58-2B-4T model + GGUF (HF): https://huggingface.co/microsoft/bitnet-b1.58-2B-4T and https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
- TerEffic — ternary LLM inference on FPGA (455 tok/s/W, LUT-based ternary core): https://arxiv.org/html/2502.16473
- TeLLMe — energy-efficient ternary LLM accelerator on **edge** FPGA (<7 W, prefill+decode): https://arxiv.org/html/2504.16266
- Speculative decoding — comprehensive survey: https://aclanthology.org/2024.findings-acl.456.pdf and https://arxiv.org/pdf/2401.07851
- Memory wall / batch-1 decode framing — XQuant: https://arxiv.org/abs/2508.10395 ; Bandwidth/Compute/Capacity: https://arxiv.org/html/2507.14397v1
- nvidia-smi power-measurement accuracy (box-car averaging caveat + mitigation): https://arxiv.org/html/2312.02741v2
- TokenPowerBench (LLM energy benchmarking methodology, J/token): https://arxiv.org/html/2512.03024v1
- Arty A7 Reference Manual (XADC 5 V current sense: 5 mΩ + INA199A1, 250 mV/A, AUX ch 1/9): https://digilent.com/reference/programmable-logic/arty-a7/reference-manual
- nvidia-smi power.draw / NVML docs: https://docs.nvidia.com/deploy/nvidia-smi/
- NVIDIA driver on Ubuntu (blacklist nouveau + initramfs + reboot): https://linuxconfig.org/how-to-disable-blacklist-nouveau-nvidia-driver-on-ubuntu-22-04-jammy-jellyfish-linux
- NVIDIA headless/server driver install (compute-only): https://oneuptime.com/blog/post/2026-03-02-install-nvidia-drivers-ubuntu-server/view
- Live driver swap without reboot (best-effort, caveated): https://bbs.archlinux.org/viewtopic.php?id=269686
- Brevitas (PyTorch QAT → ternary export): https://github.com/Xilinx/brevitas
- FINN (dataflow QNN→FPGA, baseline/alt): https://github.com/Xilinx/finn
- LiteX (SoC scaffold: LiteDRAM/LiteEth/VexRiscv on Arty): https://github.com/enjoy-digital/litex
