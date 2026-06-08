# ternfpga

**A multiplier-free, sparsity-skipping ternary LLM-inference engine on a $130 FPGA — built to beat a GPU on the axis that actually matters at the edge: energy-per-token.**

Batch-1 LLM *decode* is **memory-bandwidth-bound**: every token streams the whole weight matrix from memory once, at ~1–2 FLOP/byte, so the GPU's tensor cores sit idle and *moving bytes* is the cost. Two compounding escapes — and a GPU can do **neither** in silicon:

- **Ternary weights** (BitNet b1.58, `w ∈ {−1,0,+1}` ≈ 1.58 bit): the multiply becomes a **sign-select**, ~10× less traffic. GPUs dequantize ternary back to INT8/FP16.
- **Activation sparsity**: BitNet b1.58's squared-ReLU FFN is **~60% zero per token** (measured — [`activation_sparsity.md`](bench/results/activation_sparsity.md)); relu-fied / ProSparse variants reach **85–95%**. Those columns never need fetching — and GPUs accelerate only rigid **2:4** structured sparsity, not the per-token *unstructured* kind.

So we build **one hand-authored ternary processing element** — `acc += (w=+1 ? a : w=−1 ? −a : 0)`, a single 6-LUT, **zero DSP multipliers** — and wrap it three ways on a single Xilinx **Arty A7-35T**, benchmarked head-to-head against an **RTX 3060** in the same machine.

## The three directions (one core)

| Dir | What | Honest result vs RTX 3060 |
|---|---|---|
| **A** | Ternary energy/token engine | **~4–8× better energy/token** (loses ~10–15× on raw tok/s — *by design*) |
| **D** | Skip the unstructured sparsity GPUs waste | **~2.5× on the FFN** at the measured 60% (the **10–20×** needs relu-fication to 85–95%); skips per-token FLOPs the GPU *can't* |
| **B** | Double as the GPU's spec-decode draft engine | **~2.8× energy, ~2.7× latency** vs GPU-only |

We **concede raw throughput on purpose** and compete on **perf/watt, batch-1 latency, and a capability the GPU lacks**. No splashy "40×" headline — the defensible, must-clear claim is **4–8× / 10–20× on identical numerics**, measured board-to-board. See [`docs/BUILD-PLAN.md`](docs/BUILD-PLAN.md).

## Why it's novel
Ternary done in hardware exists (TerEffic; a full ternary BitNet on FPGA, TeLLMe — but on a ~$300 Zynq KV260, and with *no per-token sparsity*). Sparsity-on-FPGA exists (FlightLLM — on HBM datacenter parts). **Nobody has combined ternary × per-token unstructured sparsity on a sub-$150 board** ([feasibility study](docs/research/scaling-feasibility.md)). That intersection is this project.

## Layout
```
rtl/      hand-written SystemVerilog (ternary PE, sparse skip, DDR3 stream)
sim/      cocotb + verilator testbenches (TDD: tests land before RTL)
bench/    benchmark harness + results (FPGA vs RTX 3060, energy/token)
models/   ternary model quantization / relu-fication pipeline
docs/     the design dossiers (A/B/D), build plan, benchmark methodology, sources
tools/    sync-to-worker4 + build/flash helpers
```

## Status
🚧 **Phase 0** (de-risk the core in simulation). **✅ Running on real silicon** — the multiply-free ternary engine is flashed to a physical **Arty A7-35T** and verified computing **bit-exact, read back over UART** (16/16 `y==2c`, [`bench/results/onboard.md`](bench/results/onboard.md); 105 LUTs, 0 DSP, 100 MHz, **~63 mW** on-chip — ~2000× under CPU/GPU, [`bench/results/power.md`](bench/results/power.md)). Verified bit-exact in cocotb/Verilator so far: **`ternary_dot`** (multiply-free dot, 0 DSP), **`ternary_gemv`** (row-streamed matrix-vector), and **`ternary_gemv_sparse`** (activation-sparse gather — fetches only active rows; measured **50–94% weight-byte savings** at 50–6% density, [`bench/results/sparse_skip_sim.md`](bench/results/sparse_skip_sim.md)). **Energy/token head-to-head, measured** (batch-1, BitNet-2B-4T): CPU 5950X (native ternary) = **4.62 J/tok**, RTX 3060 (bf16 — *can't do ternary*, so it dequantizes) = **3.67 J/tok** and is even *slower* (23.5 vs 28.4 tok/s) ([`bench/results/gpu_baseline.md`](bench/results/gpu_baseline.md)). The GPU gets almost no benefit from the 1.58-bit weights — exactly the gap the FPGA exploits: target **~0.25–0.4 J/tok** (~10× under the 3060 on the same model) at sub-watt power. **Phase 1:** the engine runs **as a peripheral in a RISC-V SoC on the board** (LiteX VexRiscv + LiteDRAM DDR3) — the CPU drives a GEMV and reads `y` back **bit-exact** (`TERNARY_ONBOARD_PASS`, 16 rows; [`bench/results/onboard_soc_gemv.md`](bench/results/onboard_soc_gemv.md)), with 256 MB DDR3 calibrated + Memtest-OK ([`ddr3_onboard.md`](bench/results/ddr3_onboard.md)). The integrated memory→unpack→0-DSP-compute datapath is proven on silicon. **Phase 2 (scaling — de-risked):** a [feasibility study](docs/research/scaling-feasibility.md) (multi-source, adversarially verified) re-scoped the target from a *full model* (a full BitNet 0.73B does **not** fit a 35T — its ternary core alone exceeds our LUT budget; that build lives on a ~$300 KV260) down to **one real-width transformer block**, glue on the VexRiscv host. A [P&R fit sweep](bench/results/fit_sweep.md) confirms **0 DSP up to FFN width 2048** (the wall is register-resident operands → move to BRAM), and BitNet b1.58's FFN activation sparsity is now **measured at ~60%** ([data](bench/results/activation_sparsity.md)) — real and GPU-unmatchable, but below the assumed 85–95%, which needs relu-fication. **Synthesis (`xc7a35t`):** all three modules use **0 DSPs** (multiply path is pure LUT+CARRY), <2.5% LUTs, ~104–116 MHz unpipelined → **~280 MHz pipelined** (`ternary_dot_pipe`, 2.7×, still 0 DSP) ([`bench/results/utilization.md`](bench/results/utilization.md)). **Model→RTL:** real BitNet ternary weights (1bitLLM 0.7B, layer-0 `gate_proj`) run **bit-exact** through the engine via the export pipeline ([`models/`](models/)). Build log: [`BUILDLOG.md`](BUILDLOG.md) · architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Hardware / toolchain
Arty A7-35T (`xc7a35t`) on dev host `worker4` · Vivado 2025.2 + verilator + cocotb + openFPGALoader · RTX 3060 12 GB as the benchmark baseline.

## Contributing
Test-first, benchmark-early. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License
[Apache-2.0](LICENSE).
