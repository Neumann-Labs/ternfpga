# Build Log

Append-only, dated narrative of decisions, measurements, dead-ends, and surprises. Source material for the eventual write-up. Newest entries at the bottom of each day.

---

## 2026-06-08 — Genesis & the pivot

**Origin.** Started as a "what could a $130 Xilinx Arty A7 do that's genuinely impressive?" exploration. First framing chased determinism / cycle-exact matmul / a security-attestation angle. Killed it — matmul is elementary, NN training *wants* stochasticity so bit-exact determinism is a niche concern, and the security angle wasn't the point.

**The pivot.** Reframed to: *solve a real ML-systems problem the industry actually cares about, using the cheap FPGA, benchmarked against a real GPU.* Ran an 8-angle problem-space survey (32 opportunities, 119 verified sources). The signal that showed up loudest, repeatedly: **ternary (BitNet b1.58) LLM inference** and **activation sparsity** — two things a GPU structurally cannot exploit in silicon.

**Key realization — be honest about the hardware.** The Arty A7-35T has ~280× less memory bandwidth than an RTX 3060. It will **lose raw throughput**, full stop. The credible wins are **energy-per-token**, **batch-1 latency**, and **capability** (native ternary, per-token unstructured sparsity). Any claim of "40× faster" would be dishonest; the defensible claim is **4–8× better energy/token (ternary) / 10–20× (+sparsity) on identical numerics**, measured board-to-board.

**The program (A/B/D).** One hand-built ternary PE (`acc += (w=+1?a:w=−1?−a:0)`, a 6-LUT, zero DSP) wrapped three ways:
- **A** — ternary energy/token engine (DDR3-streamed ~300M model; the 2B doesn't fit 256 MB).
- **D** — skip 85–95% activation sparsity that the GPU computes anyway → cut DDR3 traffic, the actual bottleneck.
- **B** — reuse it as the 3060's speculative-decode draft engine (async drafting to hide the 100 Mb-Eth round-trip).

**Hardware confirmed.** Board is an Arty A7-**35T** (`xc7a35t`, IDCODE `0x0362d093`), flashable via openFPGALoader. worker4 toolchain installed: Vivado 2025.2 (verified), verilator 5.020, cocotb 2.0, openFPGALoader 0.12. Project lives on a dedicated 400 GB LV at `/srv/fpga` (off the prod `/srv/eval-results`). worker4 also has an **RTX 3060 12 GB** (currently on `nouveau`; NVIDIA card is the *primary* boot GPU and owns the console framebuffer → the CUDA-driver install will want a reboot in a maintenance window — deferred until the Phase-1 benchmark, since Phase 0 needs no GPU).

**Repo created.** `Neumann-Labs/ternfpga`, Apache-2.0. Practices: TDD (cocotb test before RTL), benchmark-early, Conventional Commits, this build log. Authoring on the Mac; build/sim/flash on worker4.

**Next:** Phase 0 — hand-write the ternary PE, prove it bit-exact vs a NumPy golden in cocotb, simulate the sparsity bytes-saved curve, stand up the `bitnet.cpp` CPU baseline, and pick/distill the ~300M relu-fied ternary model.

*Verified anchors:* BitNet b1.58 (arXiv 2402.17764) · TerEffic 455 tok/s/W (2502.16473) · TeLLMe <7 W (2504.16266) · ProSparse 89% sparsity (2402.13516) · batch-1 decode wall (2605.30571).

### Phase 0, cycle 1 — the ternary core is bit-exact ✅

First TDD cycle, test-first. Wrote the NumPy golden (`models/ternary_ref.py`), the cocotb test (`sim/test_ternary_dot.py`), then the RTL (`rtl/ternary_dot.sv`) — a **multiplier-free ternary dot product**: each "multiply" is a 6-LUT sign/zero select (`01=+1`, `10=-1`, `00=0`), summed in an adder tree, **zero DSP**. The dev loop works end-to-end: author on the laptop → `tools/sync_worker4.sh` rsyncs → worker4 runs Verilator + cocotb → result streams back.

**Result:** `6 directed edge cases + 2000 randomized K=8 dot products, bit-exact vs numpy, 0 mismatches.` The literal core datapath of the whole engine is verified before we build anything around it.

**Toolchain note / tech debt:** worker4 + GitHub-CI apt Verilator is **5.020**; cocotb 2.0 needs **≥5.036**, so we pinned **cocotb<2 (1.9.2)** in `requirements-dev.txt`. Works perfectly; modernizing the sim stack is filed as a tracked follow-up issue.

**Next (cycle 2):** generalize the lane into the `K`-wide / tiled **GEMV** (matrix-vector) with a parameterized adder-tree, then add the **sparse skip** path (gate lanes on a per-token active mask) and a cocotb test that measures cycles/bytes saved vs dense — Direction D's central lever, in simulation.

### Phase 0, cycle 2 — ternary GEMV + a scalable test harness ✅

Built `ternary_gemv` (`rtl/ternary_gemv.sv`): a row-streamed ternary matrix-vector multiply `y = W·x`, one weight row per cycle through the `ternary_dot` lane (still **zero DSP** in the multiply path), with results read back through a narrow `rd_addr/rd_data` port. Test-first (`tb_ternary_gemv.py`) vs the NumPy golden. **Result: 60 random K=8 / M=16 GEMVs bit-exact, 0 mismatches** (dot suite still green: 6 edge + 2000 random). Added `docs/ARCHITECTURE.md` (datapath, the int8/ternary encoding contract, module map, roadmap, resource budget).

Two debugging findings worth recording (both → the toolchain-modernization issue #1):
- **cocotb 1.9's experimental `cocotb.runner` can't bind a top module that instantiates a submodule** on Verilator 5.020 ("Can not find root handle ternary_gemv"). The classic `Makefile.sim` flow works perfectly for the same RTL — so the suite drives `Makefile.sim` once per DUT (isolated build dirs), via `sim/Makefile`.
- Verilator + cocotb VPI **double-frees on very wide (>~256-bit) public signals**; a 512-bit flat result bus tripped it. Reading results through a narrow address port fixed it — and is better hardware design (no giant result bus).

**Next (cycle 3):** the **sparse-skip** path — gate the dot lanes on a per-token active-neuron mask, add a cocotb test that *measures* cycles + weight-bytes skipped vs dense across sparsity levels (Direction D's lever, quantified in sim). Then stand up the `bitnet.cpp` CPU baseline (first real energy number — no GPU needed).

### Phase 0, cycle 3 — the sparse-skip gather engine ✅ (Direction D's lever, measured)

Built `ternary_gemv_sparse` (`rtl/ternary_gemv_sparse.sv`): an FSM-driven gather engine that, given a per-row `active_mask`, issues a weight-memory read **only for active rows** and skips inactive ones entirely (no address issued, no data fetched) — the literal mechanism by which activation sparsity cuts the DDR3 traffic that bottlenecks decode. Test-first (`tb_ternary_gemv_sparse.py`) vs the NumPy golden, with a synchronous-read weight-ROM coroutine modeling memory.

**Measured (M=16, K=8), weight bytes fetched vs dense:**

| density | rows fetched | weight bytes saved |
|---|---|---|
| 100% (dense) | 16/16 | 0% |
| 50% | 8/16 | 50% |
| 25% | 4/16 | 75% |
| 6% (1/16) | 1/16 | 93.8% |

`rows_fetched == active rows` exactly, output bit-exact. At the 85–95% activation sparsity of relu-fied/ProSparse FFNs (ProSparse Llama-2-7B = 89.3%), this fetches ~5–15% of dense weight bytes — the bandwidth reduction a GPU's dense / 2:4 MAC array cannot achieve for per-token *unstructured* sparsity. Captured in `bench/results/sparse_skip_sim.md`. Suite now: `dot` + `gemv` + `sparse`, all bit-exact.

**Next (cycle 4):** stand up the **`bitnet.cpp` CPU baseline** on worker4 — build it, run a small ternary model, capture tokens/sec + CPU energy (RAPL). The first real cross-reference number, no GPU. Then begin the model/quantization pipeline (pick + relu-fy/distill the ~300M ternary target).

### Phase 0, cycle 4 — first measured CPU baseline (bitnet.cpp) ✅

Stood up the CPU ternary-inference baseline; reproducible scripts in `bench/cpu_baseline/`. Hit (and fixed + automated) two upstream issues on Ubuntu 24.04 / clang-18:
1. `ggml-bitnet-mad.cpp:811` — `int8_t * y_col = y + ...` where `y` is `const`: a hard error under clang-18 (older clang/gcc only warn). Patched to `const int8_t *` (line 906 already was); `setup_bitnet.sh` applies the sed automatically.
2. `setup_env.py`'s HF→GGUF converter rejects the 2B-4T architecture (`BitNetForCausalLM` "not supported"). Worked around by pulling Microsoft's official **pre-quantized i2_s GGUF** (`microsoft/bitnet-b1.58-2B-4T-gguf`, 1.19 GB) instead of converting.

**Measured (BitNet b1.58 2B4T, i2_s, Ryzen 9 5950X, 16 threads, 256 tokens):**

| throughput | energy/token | avg power |
|---|---|---|
| **28.4 tok/s** | **~4.62 J/tok** (RAPL package, full run) | ~121 W |

Energy via AMD RAPL (`/sys/class/powercap/intel-rapl:0` — no `perf` needed). Captured in `bench/results/cpu_baseline.md`. This anchors the energy axis: the FPGA target is ~0.25–0.4 J/tok (~10× better than this CPU), and the head-to-head GPU (3060, ~1–2 J/tok projected) is the one step that needs the NVIDIA driver — deferred.

**Next (cycle 5):** push the RTL toward synthesis reality — run **Vivado synthesis** on `ternary_dot`, `ternary_gemv`, `ternary_gemv_sparse` to get real **LUT / DSP / BRAM utilization + Fmax** on the `xc7a35t`, confirming the **0-DSP multiply path** on actual silicon resources. Still no GPU.

### Phase 0, cycle 5 — Vivado synthesis: 0 DSPs confirmed on silicon ✅

Out-of-context Vivado 2025.2 synthesis of all three modules on the real part (`xc7a35ticsg324-1L`). Reproducible via `syn/run_synth.sh` + `syn/synth_one.tcl`.

| module | LUTs | FFs | **DSP48** | BRAM | Fmax (synth est.) |
|---|---|---|---|---|---|
| `ternary_dot` | 233 (1.1%) | 0 | **0** | 0 | comb. |
| `ternary_gemv` | 384 (1.9%) | 582 | **0** | 0 | ~104 MHz |
| `ternary_gemv_sparse` | 521 (2.5%) | 664 | **0** | 0 | ~116 MHz |

**`DSP48 = 0` on every module** — Vivado confirms the ternary multiply is pure LUT sign-select + CARRY4 adders, freeing all 90 DSPs. The central architectural claim is now *silicon-validated*, not asserted. Footprint is tiny (<2.5% LUTs). Fmax ~104–116 MHz is the **unpipelined v0** synth estimate (critical path = the K=8 adder tree, ~14 CARRY4 levels) — already in the dossiers' target range; pipelining + place-and-route are Phase 1. Captured in `bench/results/utilization.md`. (Also fixed a Tcl bug: `report_timing_summary` rejects `-delay_type` paired with `-setup`.)

**Next (cycle 6):** the model/quantization pipeline — pull a small ternary model (the 0.7B `1bitLLM/bitnet_b1_58-large`, or distill toward ~300M), export its ternary weights to the packed format the RTL consumes (matching `models/ternary_ref.py`'s encoding), and add a sim test that runs a real model layer's weights through the gather engine. Still GPU-free.

### Phase 0, cycle 6 — model→RTL export pipeline + real-weight validation ✅

Built the bridge from a trained BitNet model to the engine, and validated the RTL on REAL ternary weights (not just random):
- `models/export_weights.py` — `ternarize_absmean` (BitNet b1.58 weight quant), pack/unpack to the 2-bit RTL encoding (shared with `ternary_ref.py` — one source of truth), `save_tile`/`load_tile`. Self-test passes.
- `models/extract_bitnet_layer.py` — reads one weight tensor straight from safetensors (no model arch / `trust_remote_code`), ternarizes, slices a tile, saves `.npz`.
- `sim/tb_gemv_from_file.py` + `make -C sim real` — runs an exported tile through `ternary_gemv`, bit-exact vs numpy.

**Measured (1bitLLM/bitnet_b1_58-large, layer-0 `gate_proj`, 4096×1536):** absmean ternarization → **34.0% weight sparsity** (static zeros); the extracted 16×8 tile runs **bit-exact** through the engine. Committed `models/data/real_tile.npz` (682 B) as a test fixture so `make -C sim real` works from a clone.

Honesty note: 34% is *static weight* sparsity; Direction D's bigger lever is *activation* sparsity (85–95%, dynamic per-token), a separate runtime quantity — both are exploitable by `ternary_gemv_sparse`'s gather. `models/README.md` documents the pipeline.

**Next (cycle 7):** parallelize the engine — a `P`-lane `ternary_pe_array` (bandwidth-matched, the throughput step) and/or pipeline the K-wide adder tree to lift Fmax past the v0 ~104–116 MHz; then a weight-unpacker (5 ternary/byte dense packing) toward the DDR3 streaming path. Still GPU-free.

### Phase 0, cycle 7 — pipelined dot: Fmax 104 → 280 MHz ✅

`rtl/ternary_dot_pipe.sv` — the multiply-free dot, but the K-wide reduction is split into **3 registered stages** (sign-select → two half-sums → final sum), so no combinational path runs the whole adder tree. Streaming: a new input every cycle, 1 result/cycle, `valid_in` pipelined to `valid_out`. Test-first (`tb_ternary_dot_pipe.py`, 800 streamed dots, FIFO-tracked vs the golden): **bit-exact**.

**Synthesis (xc7a35t):** critical path drops from ~14 → **4 logic levels**; WNS +0.424 ns @4 ns → **Fmax ~280 MHz** (2.7× the ~104 MHz combinational GEMV), at **149 LUTs / 129 FF / 0 DSP** (it even shrinks — registers break the tree into cheaper pieces). The latency cost is 3 cycles, irrelevant for a streaming engine. Updated `bench/results/utilization.md`.

**Next (cycle 8):** fold the pipelined lane into the GEMV/gather datapath (a `valid`-tracked streaming GEMV at the higher clock), and/or build the **weight-unpacker** (dense 5-ternary/byte → 2-bit codes) that feeds it from a DDR3 burst — the first piece of the real memory path. Still GPU-free.
