# ffn_glue_unit — synthesis on the Arty A7-35T (#46)

Out-of-context Vivado synthesis of `rtl/ffn_glue_unit.sv` (F_MAX=6912) for `xc7a35ticsg324-1L`.

## Utilization — fits comfortably
| resource | used | avail | % |
|---|---:|---:|---:|
| Slice LUTs | **1492** | 20800 | 7.2% |
| Slice Registers | 432 | 41600 | 1.0% |
| Block RAM Tile | 20 | 50 | 40% |
| **DSPs** | **19** | 90 | 21% |

gate_int/up_int/w_q + the h_q output live in **BRAM** (20 tiles); the int64-range mults
(relu(g)² · up · w, and |N|·recip) infer **19 DSPs** — idle in the 0-DSP ternary engine.

> **The fit took one fix.** First synth: **115% LUT, 134% FF** — `hqmem` (6912×8) was built as a
> flip-flop array + 6912-deep address decoder (~55K FF, ~22K LUT) because its write (FSM block) and
> read (clock-only block) sat in *different* always blocks, breaking the BRAM template. Moving the
> write into the same clocked block as the read → simple-dual-port BRAM → **7% LUT / 1% FF**. (The
> same class of bug as the Phase-2 `y_mem` and Phase-5 `kmem` BRAM-inference fixes — now thrice-seen.)

## Timing
WNS **−42.7 ns @ 100 MHz** → Fmax ≈ 19 MHz. The critical path is the **single-cycle compute chain**
(relu → g² → ×up → ×w in pass 1; that **plus** ×recip → 128-bit barrel-shift in pass 2). This was a
deliberate simplicity-over-Fmax choice: a **~6-stage pipeline** (one multiply / the shift per stage)
closes 100 MHz at the **same cycle count**. It is **immaterial to the energy result**: the FFN glue
is now ~15.6K of a ~9.65M-cycle layer (~0.16%), so 19 vs 100 MHz shifts J/token by <0.1%. The
measured/derived number uses the **cycle count** (15,633/layer), which is timing-independent.

## Result
Bit-exact vs `models/ffn_glue_unit_ref.py` (`make -C sim ffnglue`, FFN_GLUE_UNIT_PASS), **2.26
cyc/channel → ~15.6K cyc/layer vs the measured 2.58M host FFN glue = ~165× collapse**. BRAM is the
watch item for on-board integration (20 tiles here + the SoC's ~27 > 50) — a tuned F-tile or sharing
the activation BRAMs is the integration step.

_Reproduce:_ `vivado -mode batch -source syn/synth_one.tcl -tclargs xc7a35ticsg324-1L ffn_glue_unit 10.0 rtl/ffn_glue_unit.sv`
