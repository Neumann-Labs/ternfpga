# Fully-measured end-to-end J/token — and the glue-bound verdict

Phase-4 #35–#37. Phase 3 left one gap: the host-glue cycle count was unmeasured (the firmware
trapped). The cause turned out to be a **missing `irq_setie`** — LiteX's `uart_write` is
IRQ-driven, so without interrupts the TX ring stalls ~8 chars in (the float/libm/timer theories
were all red herrings). With that fixed and the glue rewritten **pure-integer** (RoPE/softmax via
Q15 LUTs, RMSNorm/FFN via the integer-cancellation trick — no float, no libm), it runs on silicon
and times cleanly.

## Measured on silicon (one BitNet-2B layer of glue, pos=64, timer0 @ 100 MHz)
```
norm_x2 = 544,857     rope = 80,425     attn(scores+softmax+a·V) = 16,213,542     ffn = 2,583,703
GLUE_INT_PER_LAYER = 19,422,527 cycles
```
Numerics validated separately (`models/glue_fixed_ref.py`, cosine 0.999999).

## Fully-measured cycles/token (BitNet-2B, both terms measured on silicon)
| term | cycles/layer | source |
|---|---:|---|
| ternary engine (8.68 M tiles × 1 cyc/tile) | **8.68 M** | measured (#24) |
| host glue (VexRiscv) | **19.42 M** | measured (#36) |
| **layer total** | **28.10 M** | |

× 30 layers + LM-head (41.0 M engine cycles) = **884 M cycles/token** → **8.84 s/token**
(0.11 tok/s) → **J/token = 8.84 s × 0.489 W = ~4.32 J/token**.

## Verdict: the naive host-split is GLUE-BOUND (honest, and the key finding)
| | J/token | vs RTX 3060 (3.67) |
|---|---:|---:|
| **FPGA engine compute only** (measured rate) | **~1.47** | **~2.5× better** |
| **FPGA full system, host-split glue** (measured) | **~4.32** | **~1.2× WORSE** |
| host-glue overhead (the tax) | ~2.85 | — |

The 0-DSP engine is genuinely ~2.5× more energy-efficient than the GPU — **but the host-split
glue on the soft VexRiscv erases that and then some.** The glue is **2.2× the engine's cycles**,
and **83% of it is attention** (scores/softmax/a·V = 16.2 M): those loops stream the KV cache from
DRAM on a cacheless soft CPU, so they are **DRAM-latency-bound**, not compute-bound. RoPE (LUT) and
the norms are negligible; even the FFN glue (integer, 2.58 M) is small. So the bottleneck is
specifically **host-side attention over DRAM-resident KV**.

## The fix is clear (and is what the SOTA already does)
- **Put attention on the fabric** (scores/softmax/a·V as a hardware unit, KV in BRAM) — this is
  exactly why TeLLMe v2 / FlightLLM keep attention on-chip, not on a host CPU. That removes the
  16.2 M-cycle term and makes the system **engine-bound** → approaching the ~1.47 J/token engine
  figure (~2.5× better than the GPU).
- Or an **on-chip glue unit** for the norms/RoPE/requant (small, the FFN glue is already integer).
- Firmware micro-opt (block the loops, keep `qrot` resident, skip via the 60% activation sparsity)
  helps but does not beat the fundamental DRAM-latency wall for KV on a cacheless core.

## What is now truly measured vs still open
- **Measured end-to-end on silicon:** engine rate (1 cyc/tile), DDR3 roofline (1.42 GB/s), glue
  cycles (19.4 M/layer), SoC power envelope (0.489 W, Vivado-estimated) → a **fully-measured
  cycles/token and J/token** (no projection in the cycle path; only power is estimated).
- **The honest headline:** the *engine* wins (~2.5×); the *naive host-split system* does not
  (~1.2× worse) — it's glue-bound on host-side attention. The architecture lesson is decisive:
  **attention must live on the fabric.**
- **Still open:** an on-fabric attention/glue unit (to realize the engine's win at the system
  level), live power, independent reproduction, the wider-K / DMA-fed real decode loop.

_Reproduce:_ `python soc/firmware/gen_glue_luts.py > soc/firmware/glue_luts.h`; build `bench_glue_int.c`
via the plain `litex_bare_metal_demo` flow vs `build_dmabw`; flash + `litex_term`. **Keep the
`irq_setmask(0); irq_setie(1)` block** — without it the UART stalls.

---

## Phase 5 — on-fabric attention flips the verdict

The fix for the glue-bound result: move attention to the fabric. `rtl/attention_unit.sv` (KV in
BRAM, scores int16-MAC → shift+exp-LUT softmax → a·V) is **bit-exact** vs the oracle
(`ATTENTION_UNIT_PASS`) and runs at **~1 MAC/cycle**: ~165K cyc/layer (×20 heads) vs the **16.2M**
host attention — a **~98× collapse** (~49× at T=64). Synthesis fits the 35T (24% LUT, **4 DSP**,
18.5 BRAM; Fmax ~86 MHz, immaterial as attention is ~1.6% of the layer — `attention_unit_syn.md`).

Recomputed cycles/token with on-fabric attention replacing the 16.2M host term:

| | cyc/layer | J/token | vs RTX 3060 (3.67) |
|---|---:|---:|---:|
| host-split (measured) | 28.1 M | **4.32** | 1.2× **worse** |
| **+ on-fabric attention** (this work) | **12.2 M** | **~1.99** | **~1.8× better** |
| engine bound (+ FFN-glue on-fabric, future) | ~8.9 M | ~1.47 | ~2.5× better |

New glue/layer = norms 0.54M + RoPE 0.08M + FFN 2.58M (all measured) + on-fabric attention ~0.33M
(sim-measured) ≈ **3.5M** — the engine (8.68M) is now **71% of the layer**: the system is
**ENGINE-DOMINANT**, and the 0-DSP engine's energy win is realized at the system level (**~1.8×
under the GPU**, flipped from 1.2× worse). The largest remaining glue is the **FFN glue** (2.58M,
DRAM-bound on the host) — moving *it* on-fabric next approaches the ~1.47 J/token engine bound
(~2.5×). Engine + glue cycle terms are silicon-measured; the attention unit is sim-/synth-backed
(on-board SoC integration with a tuned T_MAX is the remaining step).
