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

### Phase 0, cycle 8 — dense base-3 weight packing: 1.6 bits/weight ✅

`rtl/ternary_unpack5.sv` + `models/export_weights.py` (`pack_trits5` / `unpack_trits5` / `trit_codes5` / `pack_row_trits5`): the dense storage format. 3⁵ = 243 < 256 ⇒ **5 ternary weights per byte = 1.6 bits/weight** (the log₂3 = 1.585-bit optimum; 20% tighter than the 2-bit codes, 5× tighter than INT8). Since decode is DDR3-bandwidth-bound, that 20% is a direct cut in weight traffic. The combinational decoder turns one byte into the 5 lane codes the multiply-free dots consume — so a DDR3 burst feeds the lanes directly.

Test-first, **exhaustive**: `tb_ternary_unpack5.py` checks all 243 bytes (RTL decode == Python golden, round-trip exact). **Synthesis (xc7a35t): 36 LUTs, 0 DSP, 0 CARRY** (Vivado folds the ÷3 / %3 chain into pure LUTs). Suite now: `dot` + `gemv` + `sparse` + `pipe` + `unpack`, all green.

**Next (cycle 9):** assemble the **streaming GEMV** that ties it together — DDR3-burst → `ternary_unpack5` → pipelined dot lane → accumulate, `valid`-tracked end to end at the higher clock; the first module shaped like the real decode datapath. Still GPU-free (the on-board DDR3 bring-up via LiteX/MIG is the larger follow-on).

### Phase 0, cycle 9 — packed-weight GEMV: the first real datapath integration ✅

`rtl/ternary_gemv_packed.sv` ties cycle 8 + cycle 1 together: weight rows arrive in the **dense base-3 layout** (the on-DDR3 format), a generate-array of `ternary_unpack5` decodes them to lane codes, and `ternary_dot` does the multiply-free reduction — `y = W·x` straight from packed bytes, **0 DSP** end to end. Test-first (`tb_ternary_gemv_packed.py`, K=10/M=16, rows packed via `pack_row_trits5`): **40 GEMVs bit-exact**.

**Synthesis (xc7a35t): 603 LUTs, 625 FF, 0 DSP, ~104 MHz** (WNS −5.627 ns @4 ns; the unpack is nearly free — the critical path is still the combinational K-wide adder tree, so folding in `ternary_dot_pipe` is the Fmax follow-up). This is the first module shaped like the real decode datapath (memory burst → unpack → MAC). Suite: dot + gemv + sparse + pipe + unpack + packed, all green. Updated `bench/results/utilization.md`, `docs/ARCHITECTURE.md`.

**Next (cycle 10):** the big one — start the **on-board path**: a minimal top (the ternary engine + a UART/GPIO readout) synthesized to a real bitstream and flashed to the Arty A7-35T, closing the author→synth→flash→observe loop on actual silicon. Still GPU-free; the "real hardware" milestone the repo has been building toward.

### Phase 0, cycle 10 — RUNNING ON REAL SILICON ✅

Closed the author→synth→flash→**observe** loop on the physical Arty A7-35T. `rtl/uart_tx.sv` (8N1 transmitter, sim bit-exact) + `rtl/arty_top.sv` (a `ternary_dot` of a runtime counter `c` against weights summing +2 → `y=2c`, streamed as ASCII `TN<c><y>` over the USB-UART + heartbeat LEDs). Plus `constraints/arty_a7_35.xdc`, `syn/build_bitstream.{tcl,sh}`, `syn/flash.sh` (openFPGALoader), `bench/verify_onboard.py`.

**Result: 16/16 UART lines `y == 2*c`, read off the board over `/dev/ttyUSB1`.** The multiply-free ternary engine computes correctly in fabric. Build: 105 LUTs, **0 DSP**, 100 MHz met (WNS +1.924 ns). Captured in `bench/results/onboard.md`.

The road there (honest, for the blog): UART sim-verified first. On-board, the result read `y=0` while `c` incremented correctly — a baud sweep proved the data was fine (132/133 lines decoded; the initial 0/0 was FT2232H settling), so the bug was logic. Wrote `tb_arty_top.py` (full integration sim) which **reproduced** `y=0`; probing internals showed the dot's inputs correct (`a_flat=0x0303…`, `w_flat=0xAA55`) but output 0. Root cause: **`0xAA55` packs `[+1×4,−1×4]` = sum 0**, not the intended +2; the correct constant is **`0xA955`**. Three unit tests passed but missed it — the integration test earned its keep. Fixed → rebuilt → reflashed → verified.

**Next (cycle 11):** a Vivado **`report_power`** estimate for the engine (a GPU-free energy data point), and/or begin the on-board inference datapath (DDR3/LiteX) — the larger build toward the real energy/token head-to-head, whose GPU side is the step that finally needs the RTX 3060.

### Phase 0, cycle 11 — FPGA power envelope ✅ (the energy denominator)

Vivado `report_power` (vectorless, post-route) on `arty_top` → **63 mW total on-chip** (62 mW static leakage + 1 mW dynamic; medium confidence). Reproducible: `syn/report_power.{tcl,sh}`. Captured in `bench/results/power.md`.

**The denominator of the thesis:** the xc7a35t chip draws ~0.06 W here — ~1900× under the 5950X's 121 W and ~2700× under a 3060's ~170 W. Even at the FPGA's far-lower throughput, a 3-orders-of-magnitude power gap is what makes ternary-on-FPGA win on energy/token. Honest: the demo's 1 mW dynamic is unrepresentative (it barely toggles); a busy engine lands ~0.2–0.5 W, still sub-watt. The accurate J/token needs a SAIF of the real engine + measured tokens/sec.

**Phase 0 is essentially complete.** The ternary engine is designed, **bit-exact in sim** (7 testbenches), **0-DSP-confirmed in synthesis**, validated on **real BitNet weights**, **running on real silicon** (verified over UART), and **power-profiled**. What remains before the energy/token head-to-head is (a) the on-board inference datapath — DDR3/LiteX streaming, a large build — and (b) the **GPU baseline, which needs the RTX 3060**. That is: the autonomous build has reached the point where the next decisive measurement requires the GPU.

### Phase 0+, cycle 12 — parallel PE array (the throughput knob) ✅

(Resumed past the Phase-0 checkpoint — keep building GPU-free.) `rtl/ternary_pe_array.sv`: P parallel `ternary_dot` lanes computing P output rows/cycle against a shared activation vector. Test-first (`tb_ternary_pe_array.py`, 500 trials × P=4): **bit-exact, all lanes**. (The test also caught a golden-argument-order slip in my own tb — `ternary_dot_golden(activations, weights)`, not the reverse.)

**Synthesis (xc7a35t, P=4): 931 LUTs (4.5%), 0 DSP, 208 CARRY4** — exactly 4× the single dot (233 LUTs): linear scaling, multiply-free. The 35T has LUT room for dozens of lanes, but the real ceiling is the DDR3 weight bandwidth (~0.6–0.8 GB/s), so P is sized to the roofline, not the LUT budget. Updated `bench/results/utilization.md`.

**Next (cycle 13):** fold the 280 MHz `ternary_dot_pipe` into a streaming `valid`-tracked GEMV (parallel × pipelined), then wire toward the DDR3 tile feed — the Phase-1 datapath that yields a real on-board tokens/sec. Still GPU-free.

### Phase 0+, cycle 13 — streaming pipelined GEMV: the engine runs at 280 MHz ✅

`rtl/ternary_gemv_pipe.sv`: the row-streamed GEMV driven by `ternary_dot_pipe` (3-stage) instead of the combinational dot. Rows stream 1/cycle; the 3-cycle latency is absorbed by tracking `valid_out` (k-th result = row k, order preserved) — no cycle-counting. Test-first (`tb_ternary_gemv_pipe.py`, 40 GEMVs streamed back-to-back): **bit-exact**.

**Synthesis (xc7a35t): 351 LUTs, 742 FF, 0 DSP, ~280 MHz** — 2.7× the combinational GEMV's ~104 MHz, carrying the pipelining win through to the full matrix-vector. This is the high-clock streaming shape the on-board decode datapath will use. Updated `bench/results/utilization.md`.

**Next (cycle 14):** the **weight-feed front-end** — a FIFO / double-buffer that turns a DDR3-style burst of dense base-3 bytes into the `w_row` stream (`ternary_unpack5` → `valid`), so the streaming GEMV can be fed from memory; the first concrete piece of the Phase-1 DDR3 datapath. Still GPU-free.

### Phase 1, cycle 14 — weight-feed front-end (byte burst → row stream) ✅

`rtl/weight_feed.sv`: accepts one dense base-3 weight byte/cycle (the DDR3 layout), accumulates BPR=ceil(K/5) bytes per row, unpacks via a `ternary_unpack5` array, and emits `{w_row, row_valid}` — exactly what `ternary_gemv` / `ternary_gemv_pipe` consume. Test-first (`tb_weight_feed.py`, 50 rows fed as a continuous byte stream, a monitor captures each row): **bit-exact** vs `pack_weights`.

**Synthesis (xc7a35t): 75 LUTs, 18 FF, 0 DSP, ~777 MHz** — trivial and far above the engine clock, so the byte→row bridge is never the bottleneck. The memory side of the datapath now exists; the remaining gap to a full on-board GEMV-from-memory is wiring `weight_feed → ternary_gemv_pipe` and a DDR3 (LiteDRAM/MIG) source for the bytes.

**Next (cycle 15):** integrate `weight_feed → ternary_gemv_pipe` into a single **`ternary_tile`** (GEMV directly from a base-3 byte burst, end to end in sim), then stand up **LiteX/LiteDRAM** DDR3 bring-up on the Arty to source real weight bursts — the heart of Phase 1. Still GPU-free.

### Phase 1, cycle 15 — ternary_tile: GEMV from a byte burst, end to end ✅

`rtl/ternary_tile.sv` composes the verified pieces into the in-sim memory→compute path: `wbyte` stream → `weight_feed` → `{w_row,valid}` → `ternary_gemv_pipe` → `y = W·x`. Test-first (`tb_ternary_tile.py`, 30 trials, K=10/M=16, the whole weight matrix streamed as base-3 bytes): **bit-exact**.

**Synthesis (xc7a35t): 405 LUTs (2%), 796 FF, 0 DSP, ~184 MHz** (K=10's wider half-sum stage trims Fmax from the K=8 pipe's 280 MHz — still comfortably above the 100 MHz target; an extra pipe stage recovers it if needed). This is the unit a LiteDRAM/MIG front-end will drive on-board: point it at a weight tile in DRAM, stream the burst, read `y`. Updated `bench/results/utilization.md`.

**Next (cycle 16):** begin the **LiteX/LiteDRAM DDR3 bring-up** — install LiteX on worker4, generate a minimal SoC for the Arty's MT41K128M16 DDR3, and build it. This is the large heart of Phase 1 (a multi-step subproject with real-hardware DDR3 calibration risk); still GPU-free. After it: DMA a weight tile from DRAM into `ternary_tile` on-board.

### GPU baseline + the energy/token head-to-head ✅ (maintenance window)

User opened a worker4 maintenance window to set up the GPU. Installed `nvidia-driver-580` (DKMS — **no kernel change**), wrote the nouveau blacklist, and did a **live nouveau→nvidia module swap** (nouveau refcount 0; an AMD Radeon drives the display, so the 3060 is pure compute) — so **no reboot was needed** and `tailscaled` (the only path to the box) never dropped. `nvidia-smi`: RTX 3060 12GB, driver 580.159.03, CUDA 13.0. Then a CUDA-PyTorch venv (`/srv/fpga/gpu-venv`, torch 2.6+cu124; transformers pinned `<5` — the 5.x release's model-class imports are broken). `bench/gpu_baseline/run_gpu.py` measures decode tok/s + `nvidia-smi` power → J/token.

**The head-to-head (BitNet-2B-4T, batch-1, 256 tok):**

| platform | path | tok/s | power | J/token |
|---|---|---|---|---|
| CPU 5950X | i2_s ternary | 28.4 | ~121 W | 4.62 |
| **GPU 3060** | **bf16 (dequantized)** | 23.5 | 86.4 W | **3.67** |
| FPGA Arty | ternary, 0 DSP | — | ~0.06–0.5 W | target ~0.25–0.4 |

**The thesis, measured:** the 3060 has no ternary datapath, so it dequantizes BitNet to bf16 (4.87 GB) and gets **3.67 J/tok — barely better than the CPU's native-ternary 4.62, and actually slower** (23.5 vs 28.4 tok/s). The GPU extracts almost no value from the 1.58-bit weights — the exact gap the FPGA exploits (0-DSP ternary, sub-watt). The GPU's best-foot dense run (Qwen-1.5B) is 1.82 J/tok — still ~5–7× over the FPGA target. Captured in `bench/results/gpu_baseline.md`.

This closes the **baseline triad** (CPU + GPU measured, FPGA power-profiled + silicon-verified). The remaining work to a *measured* FPGA J/token is the Phase-1 on-board DDR3 inference datapath.

### Phase 1 — DDR3 working on the board (LiteX/LiteDRAM) ✅

The hardest part of the on-board datapath, done. Built a LiteX 2026.4 SoC for the Arty A7-35T (VexRiscv + **LiteDRAM** + BIOS): `syn/litex_arty.sh`. Build snags cleared: must run from a neutral dir (the `litex/` clone shadows the `litex` package), needs the `riscv64-unknown-elf` toolchain + `meson`/`ninja` for the BIOS. Bitstream met timing (WNS +0.209 ns); flashed via openFPGALoader.

**LiteX BIOS over UART:** VexRiscv @ 100 MHz, **SDRAM 256 MiB @ 800 MT/s, read leveling calibrated (m0/m1 b03), Memtest OK.** The MT41K128M16 DDR3 is functional on this board — the memory the ternary engine will stream weights from is silicon-proven. Captured in `bench/results/ddr3_onboard.md`.

**Next:** integrate `ternary_tile` into the SoC as a CSR/DMA peripheral (stage a weight tile in DDR3 → stream the base-3 burst → read `y`) for a **measured on-board tokens/sec + J/token** — the last piece to put a real FPGA number into the head-to-head.

### Phase 1 — on-board GEMV in a RISC-V SoC ✅ (the capstone)

Integrated `ternary_tile` as a CPU-controlled CSR peripheral in a LiteX VexRiscv + LiteDRAM SoC (`soc/ternary_tile_csr.py`, `soc/ternary_arty.py`; K=8 so `x` is a clean 64-bit CSR), built + flashed it, and ran firmware (`soc/firmware/main.c`) that drives the engine from the CPU and checks the result:

```
Memtest OK
Executing booted program at 0x40000000
=== ternfpga on-board GEMV (K=8, M=16) ===
TERNARY_ONBOARD_PASS  (16 rows bit-exact vs golden)
```

The full chain — CPU writes x → streams dense base-3 weight bytes → `weight_feed` → `ternary_unpack5` → pipelined multiply-free dot → y → CPU reads back — works **on silicon, in a real SoC, bit-exact**. The hardest integration risks (DDR3 calibration, CPU↔engine CSR interface) are retired. Captured in `bench/results/onboard_soc_gemv.md`; build/run in `soc/README.md`. Snags cleared: `BaseSoC` needs `integrated_rom_size`; the 80-bit x CSR (K=10) had no C accessor → K=8 for a clean 64-bit `ternary_x_write`; `litex_term` needs a pty (`script -qfc`).

**Next (future, larger scope):** a DMA weight feed (vs CPU-streamed CSR writes) for LiteDRAM-roofline throughput, then the full transformer + decode loop on-board for a measured tokens/sec + J/token. The engine, its energy advantage (0-DSP, sub-watt), and the integrated on-board datapath are all proven now — what remains is scale, not feasibility.

### Phase 2 — deep-research gate + de-risk (re-scope to one real block) ✅

Before committing many sessions to "scale the engine," the user gated on a **deep-research pass**. A multi-agent sweep (23 primary sources, 114 claims, 25 adversarially verified [2-of-3 refute to kill], 22 confirmed — `docs/research/scaling-feasibility.md`) returned a decisive, partly humbling verdict:

- **Our 0-DSP LUT ternary core is validated SOTA.** TeLLMe v2, TerEffic, and T-MAC all store ternary `{−1,0,+1}` products in LUTs (a DSP48's 25×18 multiplier is wasted on a pass/negate/zero select). We hand-built the field-standard datapath.
- **Batch-1 decode is bandwidth-bound**, unanimously. Our ~0.7 GB/s DDR3 is ~20–25× slower than even a KV260; the high-throughput "weights-on-chip" regime (TerEffic's 16,300 tok/s on a U280 with 42 MB URAM) is categorically unavailable on a 35T (~225 KB BRAM, no URAM).
- **A full BitNet 0.73B does not fit a 35T.** The closest SOTA datapoint — TeLLMe v2: a *full* ternary BitNet 0.73B at 25 tok/s / 4.8 W — runs on a ~$300 Zynq KV260, and its ternary core *alone* is ~23k LUTs > our entire 20.8k budget. The host-split it uses (CPU glue + FPGA matmul) **validates our VexRiscv design**.

So we **re-scoped** (with the user, via an explicit decision): full-model-on-board → **one real-width transformer block**, streamed from DDR3, non-ternary glue (RMSNorm/RoPE/softmax/LM-head) on the VexRiscv host, headline = **energy/token vs the RTX 3060 on bit-exact numerics**. Nobody has built an LLM on an Artix-7-class board — that white space is the opportunity. Two de-risks followed, both done before writing block RTL:

**(1) Tiled K-accumulation GEMV** (`rtl/ternary_gemv_tiled.sv`): accumulate each output row's dot over `NT` tiles of `K` lanes (row width `KT=K·NT`) through one `ternary_dot`, so a fixed lane handles real widths. **cocotb 25/25 bit-exact** (K=8 NT=4 KT=32 M=16), 0 DSP.

**(2) P&R fit sweep** (`syn/fit_sweep.sh`, `bench/results/fit_sweep.md`): OOC synthesis from toy (KT=32) to BitNet-2B FFN width (KT=2048) on the 35T —

| width KT | LUT | %LUT | FF | %FF | DSP |
|--:|--:|--:|--:|--:|--:|
| 32 | 565 | 2.7% | 865 | 2.1% | 0 |
| 1024 | 10,234 | 49% | 24,675 | 59% | 0 |
| 2048 | 11,013 | 53% | 32,961 | **79%** | **0** |

**0 DSP holds to width 2048** — the multiply-free property proven at real scale. The wall isn't compute (K=16 lanes ≈ 1.6k LUTs); it's **register-resident operands** (flat `x_reg`+`y_mem` → 79% of FFs) and a **single-cycle flat-mux** that fails timing (63–83 MHz). Verdict: the scalable block must be **BRAM-centric and pipelined**, streaming operands sequentially. The tiled GEMV is a correctness stepping-stone, not the shippable microarchitecture.

**(3) Activation sparsity, measured** (`models/measure_activation_sparsity.py`, `bench/results/activation_sparsity.md`): the sweep found **no** published figure for BitNet b1.58's FFN sparsity — Direction D's entire premise. So we measured it: hooked all 30 `down_proj` layers of BitNet-2B-4T over diverse text → **59.8% sparse** (42–79% by depth), **not** the assumed 85–95%. A real, GPU-unmatchable ~2.5× cut on `down_proj` traffic — honest, but smaller than hoped (85–95% needs relu-fication / ProSparse). The README's Direction-D claim was corrected to match the data. *(One-time analysis on the RTX 3060 — the HF BitNet loader's lazy import works in the gpu-venv but not the cpu-venv; the FPGA path stays GPU-free.)*

**Next:** build the **BRAM-centric, pipelined FFN block** (gate/up/down + squared-ReLU), TDD against a PyTorch BitNet reference; then the ~60% activation-gather on `down_proj`; then on-board it for a measured energy/token. Feasibility is settled — the architecture is now driven by real silicon numbers, not guesses.

### Phase 2 — FFN block build (TDD, in progress)

**(a) The FFN golden, validated bit-exact vs PyTorch.** Before any RTL, pinned the *exact*
BitNet-2B-4T FFN arithmetic by probing the real model (`models/inspect_bitnet_ffn.py`):
`MLP(x) = down_proj( ffn_sub_norm( ReLU(gate(x))² · up(x) ) )`, each proj an `AutoBitLinear`
(per-token int8 absmax quant → int32 ternary matmul → dequant by `weight_scale / scale_x`). The
`ffn_sub_norm` RMSNorm before `down_proj` preserves zeros, so the 60% sparsity figure stands.
`models/ffn_ref.py` implements it in NumPy and exposes the int8 inputs + int32 outputs — the FPGA
boundary. `models/validate_ffn.py` checks it against the real PyTorch FFN on captured layer-0
activations + extracted ternary weights: **cosine 1.000000, mean abs err 9e-6, PASS**. The spec is
locked (and the golden's integer path is actually *more* exact than the model's fp matmul).

**(b) The BRAM-centric streaming GEMV — the fit-sweep fix, on silicon-grade numbers.**
`rtl/ternary_gemv_stream.sv`: activation in a BRAM read by a **sequential address** (no NT:1 mux),
the K-wide dot is the **pipelined** lane, partials accumulate NT-at-a-time into a y BRAM. cocotb
bit-exact over 6 configs (incl. odd + single-tile widths). Synthesis (`xc7a35t`, 100 MHz):
**479 LUT (2.3%), 364 FF (0.9%), 0 DSP, 10 BRAM (20%), WNS +3.5 ns (Fmax ≈ 154 MHz)** — where the
register-resident flat-mux `tiled` was 53% LUT / 79% FF and **failed timing by −5.9 ns**. Operands
moved from flip-flops to BRAM → ~20×/90× LUT/FF collapse *and* timing met, resources now
width-independent (KT/M up to 8192). Snag cleared: a RAM written inside an `always_ff` with async
reset can't infer as BRAM (Vivado tried to splay 262 144 bits into registers and errored) — the
`y_mem` write went into its own clock-only block, like `x_mem`. ([`gemv_stream.md`](bench/results/gemv_stream.md),
before/after figure `bench/plots/bram_fix.png`.)

**(c) The glue simplifies — a useful identity.** Before building the inter-projection glue
(ReLU² · elementwise · `ffn_sub_norm` · requant to int8), a derivation (`models/ffn_glue_ref.py`)
showed the down_proj int8 input `h_q` depends ONLY on the **integer** `N_i = relu(gate_int_i)² ·
up_int_i · w_i`: every per-token dequant scale **and** the RMSNorm normalizer **cancel** in the
final absmax requant (`h_q = round(N · 127/max|N|)`). Verified vs the validated `ffn_ref`: float-w
gives a **100.00% exact match (0 diff)**; **16-bit fixed-point** norm weights give **99.99% (≤1
diff)**. So the "hard" float glue (dequant + RMSNorm sqrt-divide) mostly **vanishes on-chip** — the
FPGA produces `h_q` with integer multiplies + one per-token reciprocal, and the host applies only
the final per-token *output* scale. This justifies an **on-chip glue unit** that keeps the
gate/up→down path entirely on-chip — avoiding the soft-VexRiscv round-trip the research flagged as
the latency risk.

**(d) FFN block proven end-to-end — host-split, on the real GEMV.** Engineering call: the on-chip
glue unit is a genuine optimization (the integer-only identity makes it elegant) but an intricate
multi-iteration build — 71-bit `N`, a two-pass absmax requant, a fixed-point reciprocal, ~14 BRAM
at full FFN width — and it is **not on the critical path** to the headline measured number. So we
took the **host-split** the research validated: the FPGA runs the three ternary matmuls
(`ternary_gemv_stream`), the host does the now integer-only glue. `sim/tb_ffn_block.py` drives the
real RTL GEMV three times (gate → up → down) with the host glue in Python between, and checks every
integer stage **bit-exact vs `ffn_ref`** (gate/up/down) plus the glue `h_q` vs `ffn_ref` — **PASS
over 4 trials**. The FFN block datapath is complete and simulation-validated end-to-end; the on-chip
glue is parked as a documented, designed optimization (best revisited *after* we have an on-board
baseline, so its benefit can be measured, and likely paired with FFN tiling to bound the BRAM).

**(e) On-board — the scalable engine on silicon.** Wrapped `ternary_gemv_stream` as a LiteX CSR
peripheral (`soc/ternary_gemv_csr.py`, `soc/ternary_gemv_arty.py`), built the VexRiscv + LiteDRAM
SoC bitstream, flashed the Arty, and ran firmware (`soc/firmware/main_gemv.c`) that loads the
activation into the engine BRAM, streams weight tiles, and reads `y` back:
```
Memtest OK
=== ternfpga on-board streaming GEMV (K=8 NT=4 M=16 KT=32) ===
GEMV_ONBOARD_PASS  (16 rows bit-exact vs golden)
```
**All timing constraints met @ 100 MHz; total SoC on-chip power 0.500 W** (the engine itself is
479 LUT / 0 DSP / 10 BRAM; DDR3+CPU dominate the watt). The *scalable, real-width* BRAM-centric
GEMV — the exact engine the FFN runs gate/up/down on — is now proven on silicon in a real SoC, not
just the earlier toy `ternary_tile`. Snag avoided: rather than trust the wide-CSR `.re` commit
ordering for the 64-bit `x_wdata`, used a dedicated `x_we` pulse (data stable first). Honest gap:
this is `KT=32` with **CPU-streamed weights** (not the LiteDRAM roofline) — a *measured* tok/s and
J/token needs the DMA feed at real width. ([`onboard_gemv_stream.md`](bench/results/onboard_gemv_stream.md))

**(f) Activation-sparse gather — Direction D, the column-sparse half.** The existing
`ternary_gemv_sparse` skips *output rows* (the `up_proj` case: skip rows where `gate≤0`). The
`down_proj` sparsity is in the *contraction* dim (`hq` ~60% zero), so the win is **column-sparse**:
compact the nonzero `hq` and gather only the matching `Wd` columns, then run the *unchanged* dense
stream GEMV on the shorter vector. `sim/tb_gemv_gather.py` proves it **bit-exact vs the dense
golden** across densities and measures the fetch reduction:

| activation density | bytes saved |
|---|---|
| 60% | 37.5% |
| **40.2% (BitNet measured)** | **56.2%** |
| 15% (relu-fied) | 81.2% |

At the measured BitNet ~40% active, `down_proj` fetches ~44% of the dense bytes — bit-exact, with
the engine untouched (the gather is a feed/DMA concern). This is *per-token, unstructured* sparsity
a GPU's dense / 2:4 array can't exploit. ([`down_proj_gather.md`](bench/results/down_proj_gather.md),
figure `bench/plots/gather_savings.png`.)

**Next:** the **DMA weight feed** (#24) — stream weight tiles from DDR3 into the engine at the
bandwidth roofline (vs CPU CSR writes), implementing the hardware index-compaction + column gather
this measurement isolates — then scale `KT`/`M` to real FFN width and run the full block from
firmware for a **measured on-board energy/token**, the headline number.
