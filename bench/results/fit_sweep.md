# Ternary GEMV — Arty A7-35T Place-and-Route Fit Sweep

**De-risk for the "single real-width block" scope** (`docs/research/scaling-feasibility.md`).
Question: how big a ternary GEMV actually fits the Arty A7-35T, and what breaks first?

Part `xc7a35ticsg324-1L`, out-of-context synthesis (Vivado 2025.2), 100 MHz target (10 ns).
Budget: **20,800 LUTs, 41,600 FFs, 90 DSP48, ~225 KB BRAM.** Reproduce: `bash syn/fit_sweep.sh`.

| K | NT | M | width KT | LUT | %LUT | FF | %FF | DSP | BRAM | WNS (ns) | Fmax≈ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8  | 4  | 16  | 32   | 565    | 2.7%  | 865    | 2.1%  | **0** | 0 | −2.003 | 83 MHz |
| 16 | 8  | 64  | 128  | 1,591  | 7.6%  | 3,176  | 7.6%  | **0** | 0 | −4.357 | 70 MHz |
| 16 | 64 | 512 | 1024 | 10,234 | 49.2% | 24,675 | 59.3% | **0** | 0 | −5.690 | 64 MHz |
| 32 | 32 | 512 | 1024 | 8,907  | 42.8% | 24,720 | 59.4% | **0** | 0 | −6.000 | 63 MHz |
| 32 | 64 | 512 | 2048 | 11,013 | 52.9% | 32,961 | **79.2%** | **0** | 0 | −5.931 | 63 MHz |

(`K`=lanes/cycle, `NT`=tiles/row, `KT=K·NT`=row width, `M`=output rows. Fmax≈1000/(10−WNS).)

## Findings

**1. Zero DSP holds at every scale — even BitNet-2B FFN width.** The multiply-free
`{−1,0,+1}` LUT+CARRY datapath uses **0 of 90 DSP48** from a 32-wide toy up to a 2048-wide,
512-row matrix. This is the core architectural claim, now proven across the full range, not
just the toy — exactly what the SOTA literature predicts (`scaling-feasibility.md` §1).

**2. Register-resident operands are the real ceiling — not compute.** The *compute* is cheap
(K=16 lanes = ~1,600 LUTs). What balloons is storing the whole activation row (`x_reg`, 8·KT
bits) and all output accumulators (`y_mem`, 32·M bits) in flip-flops: at real width
(KT=2048, M=512) that's **32,961 FFs = 79% of the FF budget** and 11,013 LUTs = 53%, for a
*single* GEMV. An FFN block is three such matmuls (gate/up/down) plus a LiteX SoC — they will
not coexist register-resident. **Operands must live in BRAM** (~225 KB easily holds a 2 KB
activation row + 2 KB output column; weights stream from DDR3). BRAM is at 0% here precisely
because the naive design used FFs instead — the fix is to use it.

**3. The single-cycle flat-mux microarchitecture fails timing.** WNS is negative at every
point (Fmax 63–83 MHz < 100 MHz target) and worsens as NT grows, because the stationary-
activation tile is selected with a dynamic part-select `x_reg[8·K·t_idx +: 8·K]` — a 1-of-NT
mux over K-byte slices that becomes the critical path (NT=64 → 64:1). Combined with the
single-cycle K-wide combinational dot, the path is too long. **The scalable block must
(a) pipeline the dot** — we already have `ternary_dot_pipe` at ~280 MHz — **and (b) stream
activation tiles sequentially from a BRAM**, not mux a flat register.

## Implication for the build

`ternary_gemv_tiled` did its job: it **proves the K-accumulation math bit-exact** (`sim` 25/25)
and **confirms 0-DSP at real width**. But its register-resident, single-cycle, flat-mux
microarchitecture is a *correctness stepping-stone*, not the shippable one. The FFN block
(task: gate/up/down + squared-ReLU) should be built **BRAM-centric**: activation row and output
column in BRAM, weights streamed tile-by-tile from DDR3 through the **pipelined** dot, addresses
walked sequentially. The fit sweep says that design has ample headroom on the 35T — the compute
is ~8% of LUTs at K=16 and 0 DSP; the budget is freed the moment operands leave the flip-flops.

_Full per-point Vivado logs: `/tmp/ternfpga_fit/` on worker4._
