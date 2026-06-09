# ffn_glue_unit on silicon — the FFN glue, measured on real hardware (#50)

The pipelined on-fabric FFN-glue unit (`rtl/ffn_glue_unit.sv`), wrapped as a LiteX peripheral
(`soc/ffn_glue_csr.py`, F_MAX=6912), flashed to the physical **Arty A7-35T** and run from firmware:
load gate_int / up_int / w_q per channel via CSR, pulse `start`, read back `h_q[f]` + `max|N|` and
the hardware cycle counter.

## Measured (firmware over UART, 100 MHz)
```
=== ternfpga ffn_glue_unit on silicon (F=6912) ===
loaded; running...
FFNGLUE_ONBOARD_PASS  (6912 h_q + max|N| bit-exact)
MEASURED ffn-glue cycles/layer=13974 (F=6912) vs host 2.58M -> 184x
```
- **Bit-exact on silicon**: all 6912 `h_q` (int8 down_proj inputs) and `max|N|` match
  `models/ffn_glue_unit_ref.py` exactly.
- **13974 cycles/layer** at the real BitNet-2B intermediate width (F=6912) — so this *is* the
  per-layer number, no extrapolation. (The sim's ~15.8K was a naive F=512×13.5 scaling that
  double-counted the fixed per-pass drain + divide; the real F=6912 amortizes it.)
- **184× collapse** vs the measured **2.58M** host FFN glue (`glue_measured.md`).

The FFN glue now has the full **PyTorch → sim → silicon** chain (like the engine, the FFN block,
and attention). With this, **3 of the 4 system cycle terms are silicon-measured** — engine
(1 cyc/tile), attention (16456 cyc/query), FFN glue (13974 cyc/layer) — leaving only the 2× RMSNorm
(0.54M, projected) and the fully-integrated decode loop.

## Timing honesty
Post-P&R **WNS = −1.615 ns @ 100 MHz** (the SoC + FFN-glue unit; the OOC synth was −1.9 ns — P&R is
less pessimistic). Like the attention unit (P&R −1.27 ns), the chip ran it **bit-exact at 100 MHz**
at nominal voltage/temperature. The measured cycle **count** is timing-independent. A firmware note:
`done` is a 1-cycle pulse the slow CPU poll can miss (it produced a cosmetic `TIMEOUT` on the first
run though the unit had completed correctly); the firmware now waits for `cycle_count` to freeze
(the FSM stops incrementing it on return to IDLE) — robust completion detection.

## Resource (from synth, `ffn_glue_unit_syn.md`)
8% LUT / 2% FF / 40% BRAM / 19 DSP (F_MAX=6912). FFN-glue 20 BRAM + the SoC's ~27 = 47 < 50, so it
fits a 35T alongside the full SoC (the *three* accelerators together — engine + attention + FFN
glue — would exceed 50 BRAM; that integrated loop wants tiling or a bigger board).

_Reproduce:_ `python soc/ffn_glue_arty.py --run`; `gen_testvec_ffnglue.py` + `main_ffnglue.c` via the
plain `litex_bare_metal_demo` flow; flash + `litex_term`.
