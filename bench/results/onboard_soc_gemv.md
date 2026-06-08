# On-board GEMV in a RISC-V SoC — the full datapath, verified on silicon

The capstone: the multiply-free ternary engine, integrated as a CSR peripheral in
a LiteX **VexRiscv SoC** (with **LiteDRAM DDR3**), computing a real GEMV from
CPU-supplied data and verified **bit-exact** by the CPU — on the physical Arty A7-35T.

## Result (firmware over UART, 2026-06-08)
```
Memtest OK                                   <- DDR3 calibrated
Booting from serial...
Executing booted program at 0x40000000
=== ternfpga on-board GEMV (K=8, M=16) ===
TERNARY_ONBOARD_PASS  (16 rows bit-exact vs golden)
```

## The chain — all on real hardware
1. VexRiscv CPU writes the activation vector `x` (CSR `ternary_x`, 64-bit).
2. Pulses `ternary_ctl` (start: latch x, reset).
3. Streams `M·BPR` dense base-3 weight bytes (CSR `ternary_wbyte`).
4. `weight_feed` → `ternary_unpack5` → `ternary_gemv_pipe` (3-stage pipelined,
   **multiply-free, 0 DSP**) computes `y = W·x`.
5. CPU polls `ternary_status` (done), reads `y[m]` (`ternary_rd_addr`/`rd_data`).
6. All 16 rows match the NumPy golden — **bit-exact**.

The engine computes correctly **as a peripheral in a real SoC**, fed by a real CPU,
with weights in the dense base-3 format that lives in DDR3. Build/run: `soc/README.md`.

## What this establishes — and what it doesn't (honesty)
- **Does:** the full *memory → unpack → multiply-free-dot → result* datapath works
  on silicon — integrated into a RISC-V SoC, CPU-controlled, bit-exact. 0 DSP, 100 MHz SoC,
  DDR3 calibrated. The hardest integration risks (DDR3 bring-up, CPU↔engine interface)
  are retired.
- **Doesn't (yet):** a measured **tokens/sec for a full model**. That needs the whole
  transformer + decode loop on-board (a much larger accelerator build) and a **DMA**
  weight feed — the current firmware streams weights via CSR writes (CPU-bound, *not*
  the LiteDRAM roofline). The engine's throughput (1 row/cycle @ ~184 MHz) and energy
  (0-DSP, sub-watt) are established in synthesis + `report_power`; this run proves the
  integrated datapath is *correct* on hardware. A DMA-fed streaming benchmark is the
  next step toward a measured on-board J/token to drop into `gpu_baseline.md`.
