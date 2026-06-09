# attention_unit — synthesis on the Arty A7-35T (#40)

Out-of-context Vivado synthesis of `rtl/attention_unit.sv` (D=128, T_MAX=128, EXP_N=4096) for
`xc7a35ticsg324-1L`, 100 MHz target.

## Utilization
| resource | used | avail | % |
|---|---:|---:|---:|
| Slice LUTs | **4929** | 20800 | 23.7% |
| Slice Registers | 14677 | 41600 | 35.3% |
| **Block RAM Tile** | **18.5** | 50 | 37% |
| **DSPs** | **4** | 90 | 4.4% |

The KV cache (kmem+vmem, T_MAX×D×16b each) + the exp LUT live in **BRAM** (the `ram_style="block"`
attr + clock-only read block — an async-reset read block first forced LUTRAM and 90% LUTs, the same
class of bug as the Phase-2 `y_mem` fix). The int16×int16 score / a·V MACs infer **4 DSPs** — the
35T has 90, all idle in the 0-DSP ternary engine, so attention's multiplies are free real estate.

## Timing
WNS **−1.665 ns** at the 10 ns (100 MHz) target → **Fmax ≈ 86 MHz**. The critical path is the score
MAC's `acc + q*k` feeding the `> max_s` compare in one cycle; **one pipeline register** on that
path closes 100 MHz. This is **immaterial to the energy result**: attention is now ~165K of a
~12M-cycle layer (~1.6%), so 86 vs 100 MHz shifts J/token by <0.1%.

## Fit verdict
The unit fits the 35T comfortably (24% LUT, 4 DSP). **BRAM is the watch item**: 18.5 tiles here +
the SoC's ~27 (engine + DDR3 + VexRiscv) would exceed 50 — so an on-board integration would set
**T_MAX≈64** (KV depth 64 → ~10 BRAM) or share the activation BRAMs. The OOC fit + bit-exact sim
(`ATTENTION_UNIT_PASS`) + the ~98× cycle collapse establish the architecture; SoC integration (with
a tuned T_MAX) is the remaining on-board step.

_Reproduce:_ `vivado -mode batch -source syn/synth_one.tcl -tclargs xc7a35ticsg324-1L attention_unit 10.0 rtl/attention_unit.sv`
