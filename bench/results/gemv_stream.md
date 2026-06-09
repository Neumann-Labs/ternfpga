# BRAM-centric streaming ternary GEMV (`ternary_gemv_stream`)

The scalable foundation for the FFN block — the fit sweep's verdict, realized. Replaces the
register-resident, flat-part-select `ternary_gemv_tiled` (which blew flip-flops to 79% of the
35T and failed timing at real width, [`fit_sweep.md`](fit_sweep.md)) with a BRAM-resident,
sequential-address, pipelined datapath.

## Design (`rtl/ternary_gemv_stream.sv`)
- **Activation in BRAM** (K int8 per word), read one tile per cycle by a **sequential address** —
  no NT:1 mux (that mux was the flat-design's critical path).
- K-wide **multiply-free dot** = the 3-stage pipelined lane (`ternary_dot_pipe`), **0 DSP**.
- Partials accumulate **NT-at-a-time into a y BRAM**. Weight tiles stream row-major; the
  activation is loaded once and reused for every output row.
- Runtime dims `nt` (tiles/row, KT=K·nt) and `m_rows`; K=16 fixed. Supports KT and M up to 8192.

## Verification (cocotb, K=16) — `sim/tb_ternary_gemv_stream.py`
Bit-exact vs `ternary_gemv_golden` over 6 configs incl. odd and single-tile widths:
`(nt,m)` = (1,4),(4,8),(8,16),(16,32),(13,7),(32,5). **PASS.**

## Synthesis (`xc7a35t`, 100 MHz target) — the fit-sweep fix, confirmed

| metric | register-resident `tiled` @ width 2048 | **BRAM-centric `stream`** |
|---|---:|---:|
| LUT | 11,013 (53%) | **479 (2.3%)** |
| FF | 32,961 (79%) | **364 (0.9%)** |
| DSP | 0 | **0** |
| BRAM | 0 | **10 RAMB36 (20%)** |
| WNS @ 100 MHz | **−5.9 ns — FAILS** (~63 MHz) | **+3.5 ns — PASS** (Fmax ≈ 154 MHz) |

The flat-mux design's resources *grew with width* and it could not close timing; the BRAM-centric
design's resources are **width-independent** (these fixed numbers cover KT/M up to 8192) and it
**meets 100 MHz with 3.5 ns slack**. Operands moved from flip-flops to BRAM, exactly as the fit
sweep mandated — a ~20×/90× LUT/FF collapse and a timing fix in one move. This is the GEMV the
FFN block (gate/up/down) is built on.

_Reproduce:_ `make -C sim stream` · synth: `syn/run_synth.sh` (includes `ternary_gemv_stream`).
