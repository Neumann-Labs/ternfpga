# FPGA power envelope — Arty A7-35T (`xc7a35t`)

Vectorless post-route power estimate (Vivado `report_power`) for `arty_top`.
Reproduce: `bash syn/report_power.sh`.

| | W |
|---|---|
| Total on-chip power | **0.063** |
| Dynamic | 0.001 |
| Device static (leakage) | 0.062 |

Confidence: Medium (vectorless / default switching).

**The denominator of the energy thesis.** The FPGA *chip* draws ~63 mW here —
**~1900× less than the Ryzen 9 5950X's ~121 W** ([`cpu_baseline.md`](cpu_baseline.md))
and **~2700× less than an RTX 3060's ~170 W TDP**. Even at the FPGA's far lower
throughput, a power gap of this magnitude is what lets ternary-on-FPGA win on
**energy per token**.

**Honest caveats.**
- This is the **demo** (`arty_top`: one dot lane + UART), and its **dynamic**
  power (1 mW) is *not* representative — the demo barely toggles. Static leakage
  (62 mW) dominates and is the floor for any `xc7a35t` design.
- A fully-utilized ternary engine at 100 MHz adds real dynamic power, but a busy
  small design on the 35T still lands ~0.2–0.5 W; the board (FT2232H, regulators,
  DDR3) adds more — the *chip* is what's reported here.
- Vectorless = default toggle rates. A SAIF from a gate-level sim of the real
  engine running a model gives the accurate dynamic figure; that, plus measured
  tokens/sec, yields J/token.

So: **throughput is not yet measured, but the power envelope is — sub-watt, three
orders of magnitude under CPU/GPU.** Closing the energy/token comparison needs the
on-board inference datapath (DDR3 streaming) and the **RTX 3060 baseline (the GPU)**.
