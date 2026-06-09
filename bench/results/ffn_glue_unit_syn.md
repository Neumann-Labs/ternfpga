# ffn_glue_unit — synthesis on the Arty A7-35T (#46, pipelined #48)

Out-of-context Vivado synthesis of `rtl/ffn_glue_unit.sv` (F_MAX=6912) for `xc7a35ticsg324-1L`.

## Utilization — fits comfortably
| resource | used | avail | % |
|---|---:|---:|---:|
| Slice LUTs | **1719** | 20800 | 8.3% |
| Slice Registers | 887 | 41600 | 2.1% |
| Block RAM Tile | 20 | 50 | 40% |
| **DSPs** | **19** | 90 | 21% |

gate_int/up_int/w_q + the h_q output live in **BRAM** (20 tiles); the int64-range mults
(relu(g)² · up · w, and |N|·recip) infer **19 DSPs** — idle in the 0-DSP ternary engine.

> **The fit took one fix.** First synth: **115% LUT, 134% FF** — `hqmem` (6912×8) was built as a
> flip-flop array + 6912-deep decoder because its write (FSM block) and read (clock-only block) sat
> in *different* always blocks. Same-block write+read → simple-dual-port BRAM → **8% LUT / 2% FF**.
> (Same class as the Phase-2 `y_mem` and Phase-5 `kmem` BRAM-inference fixes — thrice-seen now.)

## Timing — pipelined to the silicon-proven margin (#48)
The first (single-cycle-compute) version was **WNS −42.7 ns @ 100 MHz**. Pipelining the compute into
**8 stages** (one multiply / add / shift per stage; per-stage valid + channel index + operands
travel together) closed it in three steps:

| version | WNS @ 100 MHz | critical path |
|---|---:|---|
| single-cycle compute | −42.7 ns | relu→g²→×up→×w→×recip→add→barrel-shift→clip, all one cycle |
| 8-stage pipeline | −8.9 ns | `amaxN → msb-encoder → Rsh → barrel-shift` (per-cycle) |
| + registered requant scale | −3.2 ns | `pq6 → add half → barrel-shift → clip` |
| + add/shift split | **−1.9 ns** | the `|N|·recip` DSP multiply |

**−1.9 ns OOC** matches the **attention unit's −1.7 ns** (which P&R'd to −1.27 ns and ran **bit-exact
at 100 MHz** at nominal conditions). So the on-board run is expected to work; the residual is the DSP
multiply, and the **cycle count is timing-independent** regardless. Pipelining preserved the
**bit-exactness and the cycle count** (1174 for F=512 vs 1158 single-cycle — the extra 16 is pipeline
drain).

## Result
Bit-exact vs `models/ffn_glue_unit_ref.py` (`make -C sim ffnglue`, FFN_GLUE_UNIT_PASS), **2.29
cyc/channel → ~15.8K cyc/layer vs the measured 2.58M host FFN glue = ~162× collapse**. BRAM watch for
integration: 20 tiles + the SoC's ~27 = 47 < 50 → the FFN-glue peripheral **fits a 35T alongside the
full SoC** (unlike all three accelerators at once).

_Reproduce:_ `vivado -mode batch -source syn/synth_one.tcl -tclargs xc7a35ticsg324-1L ffn_glue_unit 10.0 rtl/ffn_glue_unit.sv`
