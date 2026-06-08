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
