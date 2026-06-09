# engine + ffn_glue co-resident on silicon — the integrated capstone (#53)

Two custom accelerators — the multiply-free **ternary engine** (`ternary_gemv_stream`) and the
**FFN-glue unit** (`ffn_glue_unit`, full F_MAX=6912) — instantiated in **one** LiteX SoC
(`soc/coresident_arty.py`), flashed to the physical **Arty A7-35T**, and run **cooperating**: the
engine computes the gate/up ternary GEMVs, its int32 outputs feed the FFN-glue unit
(relu²·up·w + int8 requant), and `h_q` is read back — bit-exact, end-to-end, on the board.

## Measured (firmware over UART, 100 MHz)
```
=== ternfpga engine+ffn_glue co-resident (hidden=64 INTER=32) ===
engine gate/up GEMV: ok (0 row mismatches)
COMBINED_ONBOARD_PASS  (engine -> ffn_glue, 32 h_q bit-exact, ffn-glue 214 cyc)
```
- The engine's gate/up GEMV outputs match the golden (0/32 row mismatches), and the **end-to-end**
  `h_q` (engine GEMV → FFN-glue requant) is **bit-exact** vs the NumPy oracle.
- **First multi-accelerator computation on the board** — every prior on-silicon result measured one
  unit in isolation; this is two custom accelerators **co-resident and handing off data** in real
  hardware.

## Fit — and the honest frontier
| resource | used | avail | % |
|---|---:|---:|---:|
| Slice LUTs | 6449 | 20800 | 31% |
| Slice Registers | 4996 | 41600 | 12% |
| **Block RAM Tile** | **45** | 50 | **90%** |
| DSPs | 23 | 90 | 26% |

The ternary engine + the **full-width** FFN-glue unit (6912) + the VexRiscv/DDR3 SoC fit a 35T at
**45/50 BRAM**. This is the concrete proof of the frontier flagged since Phase 7: a *pair* fits
(45 BRAM), but adding the attention unit (~18 BRAM) → **63 > 50** — the **full three-accelerator
decode loop does not fit a single 35T**. The integrated full-token loop therefore needs **F-tiling**
(a narrower FFN-glue/attention) or a **bigger board** (Arty A7-100T ~$250 / KV260). The energy
result is cycle-count-based and holds regardless; what this phase adds is the *silicon proof that
the accelerators co-reside and cooperate* — the engine→glue handoff works on real hardware.

## Timing
Post-P&R **WNS −1.713 ns @ 100 MHz** (the SoC + both accelerators) — the same margin as the
standalone attention (−1.27 ns) and FFN-glue (−1.615 ns) units, all of which ran **bit-exact at
100 MHz** at nominal conditions, as this one did.

_Reproduce:_ `python soc/coresident_arty.py --run`; `gen_testvec_coresident.py` + `main_coresident.c`
via the plain `litex_bare_metal_demo` flow; flash + `litex_term`.
