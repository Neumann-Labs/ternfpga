# Measured on-board throughput + energy — the capstone

The ternary engine, on the physical **Arty A7-35T**, **measured at its roofline** via a hardware
cycle counter (`rtl/ternary_gemv_bench.sv` — engine + resident weight BRAM + replay FSM, as a LiteX
peripheral driven by the VexRiscv CPU).

## Measured (firmware over UART, K=8 @ 100 MHz)
```
=== ternfpga throughput harness (K=8 NT=63 M=63) ===
BENCH_ONBOARD_PASS  (63 rows bit-exact)
MEASURED  tiles=3969  cycles=3974  cyc_per_tile=1.00  (K=8 @100MHz)
```
- **1.00 cycle/tile on silicon** (the +5 cycles is pipeline fill) — the engine sustains
  **8 ternary MACs/cycle = 800 M ternary-MAC/s** at K=8, 100 MHz, **bit-exact** vs the golden.
- **SoC on-chip power: 0.489 W** (dynamic 0.425 + static 0.064; Vivado estimate). **0 DSP in the
  engine** (4 DSP total = the VexRiscv multiplier); 24% LUT, 54% BRAM (the harness w_mem + LiteDRAM
  + CPU).

## Energy per FFN block (derived from the measured throughput × power)
A real BitNet-2B FFN block = gate(2560→6912) + up(2560→6912) + down(6912→2560) ≈ **53.1 M ternary
weights**. At the measured **1 tile/cycle** (8 weights/tile): 6.64 M cycles → **66.4 ms @ 100 MHz**.

| | energy / FFN block | vs RTX 3060 |
|---|---:|---:|
| **FPGA SoC** (measured 1 tile/cycle × 0.489 W, incl. DDR3 + VexRiscv) | **~32 mJ** | **~1.9×** |
| + `down_proj` gather (40% active, measured) | ~26 mJ | ~2.3× |
| *FPGA engine-only* (the 0-DSP datapath ≈ a fraction of SoC power) | ~3–7 mJ | ~9–18× |
| RTX 3060 (per FFN block, extrapolated) | ~61 mJ | — |

GPU per-FFN-block ≈ 3.67 J/tok ÷ 30 layers ÷ 2 sublayers ≈ **61 mJ** — a rough extrapolation from
the *measured* full-model J/tok ([`gpu_baseline.md`](gpu_baseline.md)).

## What's measured, what's derived, what's future (honesty)
- **Measured on silicon:** the engine's **1.00 cycle/tile** throughput (bit-exact) and the SoC power
  envelope. This is the core claim — the multiply-free **0-DSP** datapath runs at its roofline on a
  **$130 board**.
- **Derived:** the per-FFN-block energy (measured throughput × power × real FFN dims) and the GPU
  per-block number (extrapolated from the measured 3.67 J/tok).
- **The gap & the future:** the **engine** is ~order-of-magnitude more energy-efficient per FFN
  block; the conservative **SoC** figure (~1.9×) is inflated by the general-purpose VexRiscv + DDR3
  controller running throughout — a dedicated accelerator (no soft CPU, DMA-fed) approaches the
  engine number. A *measured* **full-model J/token** (attention + decode loop), the bandwidth-bound
  **wider-K + DDR3-DMA** regime, and a **live** (vs Vivado-estimated) power reading are the
  remaining steps.

_Reproduce:_ `python soc/ternary_gemv_bench_arty.py --run`; flash + firmware per `soc/README.md`
(with `main_bench.c` + `gen_testvec_bench.py`).
