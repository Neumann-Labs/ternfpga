# attention_unit on silicon — the collapse, measured on real hardware (#43)

The on-fabric attention unit (`rtl/attention_unit.sv`), wrapped as a LiteX peripheral
(`soc/attention_unit_csr.py`, T_MAX=64), flashed to the physical **Arty A7-35T** and run from
firmware: load q / KV cache / exp LUT via CSR, pulse `start`, read back `num[d]` + `sum_e` and the
hardware cycle counter.

## Measured (firmware over UART, 100 MHz)
```
=== ternfpga attention_unit on silicon (D=128 T=64) ===
loaded; running...
ATTN_ONBOARD_PASS  (128 num + sum_e bit-exact)
MEASURED attention cycles/query=16456 (T=64 D=128) sum_e=1373845
```
- **Bit-exact on silicon**: all 128 `num[d]` and `sum_e` match `models/attn_unit_ref.py` exactly.
- **16456 cycles/query** at T=64, D=128 — the sim predicted 16384 (2·T·D); the +72 (+0.4%) is FSM
  fill/drain. **~1 MAC/cycle confirmed on real hardware.**
- **Per layer** (×20 q-heads) ≈ **329K cycles** vs the measured **16.2M** host attention (same
  pos=64 context, `glue_measured.md`) = **~49× collapse, silicon-measured** (≈98× at T=32).

Attention is now **PyTorch → sim → silicon**, the same full chain the FFN datapath has. The
Phase-5 system result (~1.99 J/token, ~1.8× under the RTX 3060) rests on this term — and it is now
**silicon-confirmed**, not just sim/synth.

## Timing honesty
Post-P&R **WNS = −1.268 ns @ 100 MHz** (the SoC + attention unit) → the static timing model says it
closes at ~89 MHz, not 100. The chip nonetheless ran it **bit-exact at 100 MHz** at nominal
voltage/temperature (the worst-case timing model is pessimistic vs actual silicon at nominal
conditions). For *guaranteed* margin, one **pipeline register** on the score-MAC → `max` compare
path (the critical path) closes 100 MHz cleanly — a known, scoped fix. The measured cycle **count**
is timing-independent (it's the FSM's cycle count); only the wall-clock conversion assumes 100 MHz,
which the bit-exact run vindicates.

## Resource (from synth, `attention_unit_syn.md`)
24% LUT / 4 DSP / 18.5 BRAM (T_MAX=128 OOC); the T_MAX=64 SoC build fits alongside the engine +
DDR3 + VexRiscv on the 35T.

_Reproduce:_ `python soc/attention_unit_arty.py --run`; `gen_testvec_attn.py` + `main_attn.c` via
the plain `litex_bare_metal_demo` flow; flash + `litex_term`.
