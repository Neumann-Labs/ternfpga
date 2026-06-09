# On-board streaming GEMV — the scalable engine, verified on silicon

The BRAM-centric streaming GEMV (`ternary_gemv_stream`, the engine the FFN block's gate/up/down
projections run on) integrated as a CSR peripheral in a LiteX **VexRiscv + LiteDRAM** SoC and
verified **bit-exact on the physical Arty A7-35T** — the Phase-1 capstone re-done with the
*scalable, real-width* engine (vs the earlier toy `ternary_tile`).

## Result (firmware over UART, 2026-06-09)
```
Memtest OK                                   <- DDR3 calibrated
Booting from serial...
Executing booted program at 0x40000000
=== ternfpga on-board streaming GEMV (K=8 NT=4 M=16 KT=32) ===
GEMV_ONBOARD_PASS  (16 rows bit-exact vs golden)
```

## The chain — all on real hardware
1. VexRiscv loads the activation into the engine BRAM (CSRs `x_waddr`/`x_wdata`/`x_we`, K int8/word).
2. Configures `nt`/`m_rows`, pulses `ctl` (start).
3. Streams `M·NT` weight tiles (CSR `w_tile`, K 2-bit codes each, row-major).
4. `ternary_gemv_stream`: sequential BRAM activation read → pipelined multiply-free dot (**0 DSP**)
   → NT-accumulate → y BRAM.
5. CPU polls `status` (done), reads `y[m]` (`rd_addr`/`rd_data`) — all 16 rows **bit-exact** vs the
   NumPy golden.

## Silicon facts
- **Timing:** *all user-specified timing constraints met* @ 100 MHz (full SoC).
- **Power:** total on-chip **0.500 W** (dynamic 0.435, static 0.064) — the **whole SoC** (VexRiscv +
  LiteDRAM DDR3 PHY/controller + the engine); DDR3 + CPU dominate. The ternary engine itself is
  **479 LUT / 0 DSP / 10 BRAM** ([`gemv_stream.md`](gemv_stream.md)), sub-100 mW class
  ([`power.md`](power.md)).

## What this establishes — and what it doesn't (honesty)
- **Does:** the *scalable, real-width* BRAM-centric streaming GEMV — the exact engine the FFN block
  runs gate/up/down on — works **on silicon, in a real RISC-V SoC, bit-exact**, meeting 100 MHz,
  0 DSP. Combined with `tb_ffn_block` (the FFN = this GEMV ×3 + host glue, sim-validated) and the
  cosine-1.0 `ffn_ref` vs PyTorch, the FFN datapath is proven from PyTorch down to silicon.
- **Doesn't (yet):** a measured **tokens/sec / J-per-token for a real-width FFN**. This run uses a
  small `KT=32` vector and **CPU-streamed weights via CSR** (CPU-bound — *not* the LiteDRAM
  roofline). A measured energy/token needs the **DMA weight feed** (#24) at real FFN width plus the
  host-glue firmware running the full block. Next: DMA + scale `KT`/`M`, then measure and drop the
  number into `gpu_baseline.md`'s head-to-head.

_Reproduce:_ build `python soc/ternary_gemv_arty.py --run`; flash + firmware per `soc/README.md`
(with `main_gemv.c` + `gen_testvec_gemv.py`).
