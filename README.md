# ternfpga

**A multiplier-free, sparsity-skipping ternary LLM-inference engine on a $130 FPGA — built to beat a GPU on the axis that actually matters at the edge: energy-per-token.**

Batch-1 LLM *decode* is **memory-bandwidth-bound**: every token streams the whole weight matrix from memory once, at ~1–2 FLOP/byte, so the GPU's tensor cores sit idle and *moving bytes* is the cost. Two compounding escapes — and a GPU can do **neither** in silicon:

- **Ternary weights** (BitNet b1.58, `w ∈ {−1,0,+1}` ≈ 1.58 bit): the multiply becomes a **sign-select**, ~10× less traffic. GPUs dequantize ternary back to INT8/FP16.
- **Activation sparsity** (relu-fied / ProSparse FFNs are **85–95% zero per token**): those weights never need fetching. GPUs accelerate only rigid **2:4** structured sparsity, not per-token *unstructured* sparsity.

So we build **one hand-authored ternary processing element** — `acc += (w=+1 ? a : w=−1 ? −a : 0)`, a single 6-LUT, **zero DSP multipliers** — and wrap it three ways on a single Xilinx **Arty A7-35T**, benchmarked head-to-head against an **RTX 3060** in the same machine.

## The three directions (one core)

| Dir | What | Honest result vs RTX 3060 |
|---|---|---|
| **A** | Ternary energy/token engine | **~4–8× better energy/token** (loses ~10–15× on raw tok/s — *by design*) |
| **D** | Skip the unstructured sparsity GPUs waste | **~10–20× better energy/token**; fetches ~15% of the bytes; ~85% FLOPs-skipped the GPU *can't* |
| **B** | Double as the GPU's spec-decode draft engine | **~2.8× energy, ~2.7× latency** vs GPU-only |

We **concede raw throughput on purpose** and compete on **perf/watt, batch-1 latency, and a capability the GPU lacks**. No splashy "40×" headline — the defensible, must-clear claim is **4–8× / 10–20× on identical numerics**, measured board-to-board. See [`docs/BUILD-PLAN.md`](docs/BUILD-PLAN.md).

## Why it's novel
Ternary done in hardware exists (TerEffic, 455 tok/s/W — but *explicitly no sparsity*). Sparsity-on-FPGA exists (FlightLLM — on a $10k Alveo). **Nobody has combined ternary × per-token unstructured sparsity on a sub-$150 board.** That intersection is this project.

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
🚧 **Phase 0** (de-risk the core in simulation). Verified bit-exact in cocotb/Verilator so far: **`ternary_dot`** (multiply-free dot, 0 DSP), **`ternary_gemv`** (row-streamed matrix-vector), and **`ternary_gemv_sparse`** (activation-sparse gather — fetches only active rows; measured **50–94% weight-byte savings** at 50–6% density, [`bench/results/sparse_skip_sim.md`](bench/results/sparse_skip_sim.md)). **First measured baseline:** BitNet 2B4T (i2_s) on the Ryzen 9 5950X CPU = **28.4 tok/s, ~4.6 J/token** ([`bench/results/cpu_baseline.md`](bench/results/cpu_baseline.md)) — the energy anchor the FPGA aims to beat ~10×. Build log: [`BUILDLOG.md`](BUILDLOG.md) · architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Hardware / toolchain
Arty A7-35T (`xc7a35t`) on dev host `worker4` · Vivado 2025.2 + verilator + cocotb + openFPGALoader · RTX 3060 12 GB as the benchmark baseline.

## Contributing
Test-first, benchmark-early. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License
[Apache-2.0](LICENSE).
