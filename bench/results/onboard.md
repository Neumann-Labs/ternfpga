# On-board verification — ternary engine on a real Arty A7-35T

The author→synth→flash→**observe** loop, closed on physical silicon. `arty_top`
(`rtl/arty_top.sv`) runs a `ternary_dot` of a runtime counter `c` against weights
summing to +2 and streams the result over the USB-UART; the host reads it and
confirms `y == 2*c`.

Reproduce (on worker4, board attached):
```bash
bash syn/build_bitstream.sh                                   # synth + P&R + bitstream
bash syn/flash.sh                                             # openFPGALoader -> Arty A7-35T
python bench/verify_onboard.py --port /dev/ttyUSB1 --n 16     # read UART, assert y==2c
```

**Result (2026-06-08): 16/16 UART lines verified `y == 2*c`** on the physical
`xc7a35t`. Sample:

| c | y (read off the board) | expected |
|---|---|---|
| 5  | 10 | 10 |
| 10 | 20 | 20 |
| 15 | 30 | 30 |
| 20 | 40 | 40 |

**Implementation:** 105 LUTs, **0 DSP**, 100 MHz timing met (WNS +1.924 ns). The
multiply-free ternary dot computes in fabric and is bit-exact with the Python
golden — the core datapath proven on real hardware before the larger DDR3 /
full-decode build-out.

**Honest scope.** This is a *compute-correctness* proof on silicon (the engine
runs and computes right), **not yet an energy measurement** — `arty_top` is a demo
harness, not full model inference. The energy/token head-to-head vs the RTX 3060
comes after the on-board inference datapath (DDR3 streaming) and the GPU baseline.

**Debug note (for the blog).** The first flash reported `y=0` while `c` incremented
correctly. A baud sweep showed the UART data was fine (132/133 lines decoded — the
initial 0/0 was just FT2232H buffer settling). The new `sim/tb_arty_top.py`
integration test then *reproduced* `y=0`; probing internals showed the dot's inputs
were correct (`a_flat=0x0303…`, `w_flat=0xAA55`) but its output was 0 — because
`0xAA55` packs `[+1,+1,+1,+1,-1,-1,-1,-1]` (**sum 0**), not the intended +2. The
correct `[+1×5,-1×3]` is **`0xA955`**. Three unit tests passed but missed it; the
integration test earned its keep. Lesson: build packed constants with
`pack_weights`, and integration-test the top — not just the units.
