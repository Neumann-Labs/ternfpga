# Architecture

How the engine is put together, why, and where each piece is going. Pairs with
[`BUILD-PLAN.md`](BUILD-PLAN.md) (the phased roadmap) and the direction dossiers
([`A`](A-ternary-engine.md) / [`D`](D-sparsity.md) / [`B`](B-wingman-specdecode.md)).

## The one idea

Batch-1 LLM decode is **memory-bandwidth-bound**: every token streams the whole
weight matrix from memory once. So the cost is *bytes moved* and *joules per
byte*, not FLOPs. We attack both with hardware a GPU can't be:

1. **Ternary weights** (`w ∈ {−1,0,+1}`): the multiply is a **sign-select**, not
   a real multiplier — and the weight is ~1.6 bits, not 16.
2. **Activation sparsity**: skip the 85–95% of FFN weight columns whose
   activation is zero this token — *don't even fetch them*.

Everything in `rtl/` is one reusable **multiply-free, sparse-skipping** compute
core, specialised three ways (energy engine / sparse engine / draft engine).

## Number encoding (the contract between RTL, golden, and model export)

| Quantity | Encoding |
|---|---|
| Activation | signed **int8**, two's complement |
| Ternary weight | **2-bit code**: `01 = +1`, `10 = −1`, `00 = 0` (`11` reserved/unused) |
| Bus packing | **little-endian**: lane/element `i` occupies bits `[w*i +: w]` |
| Accumulator / output | signed **int32** (headroom: `K=8` lanes × `127×1` ≪ 2³¹) |

The Python golden ([`models/ternary_ref.py`](../models/ternary_ref.py)) defines
this encoding once (`pack_activations`, `pack_weights`, `weight_code`); RTL and
the eventual model-export pipeline must both match it bit-for-bit.

## Module hierarchy (current)

```
ternary_dot           combinational K-wide multiply-free dot product (0 DSP)
  ├─► ternary_gemv           row-streamed matrix-vector (y = W·x)
  ├─► ternary_gemv_sparse    active-mask gather (fetch only active rows)
  └─► ternary_gemv_packed ──► ternary_unpack5   (GEMV from dense base-3 weights)
ternary_dot_pipe      3-stage pipelined dot (~280 MHz, streams 1 result/cycle)
ternary_unpack5       dense base-3 byte → 5 ternary codes (1.6 bits/weight)
```

- **`ternary_dot` (`rtl/ternary_dot.sv`)** — the literal core. For each lane,
  `+a / 0 / −a` via a 6-LUT sign-select; summed in an adder tree. **Zero DSP.**
  Combinational and fully verified bit-exact (`tb_ternary_dot.py`).
- **`ternary_gemv` (`rtl/ternary_gemv.sv`)** — streams an `M×K` weight matrix one
  row per cycle against a stationary activation vector. v0 uses a single dot
  lane (throughput is deliberately *not* the v0 goal — correctness is).
- **`ternary_gemv_sparse` (`rtl/ternary_gemv_sparse.sv`)** — the gather engine
  (Direction D). An FSM walks a per-row `active_mask` and issues a weight-memory
  read **only for active rows**, computing their dot and leaving inactive rows
  zero. The `rows_fetched` output proves weight traffic scales with density —
  verified + measured in `tb_ternary_gemv_sparse.py` (50% / 75% / 93.8% weight
  bytes saved at 50% / 25% / 6% density; see `bench/results/sparse_skip_sim.md`).
  On-silicon DDR3 gather is Phase 1.
- **`ternary_dot_pipe` (`rtl/ternary_dot_pipe.sv`)** — the dot, but split into 3
  registered stages so no path runs the whole adder tree: critical path 4 (vs ~14)
  logic levels, **~280 MHz** (2.7×), still 0 DSP, 1 result/cycle (`valid`-tracked).
- **`ternary_unpack5` (`rtl/ternary_unpack5.sv`)** — dense base-3 decoder: one byte
  → 5 ternary lane codes (`3⁵=243<256` ⇒ **1.6 bits/weight**, the log₂3 optimum;
  20% tighter than 2-bit codes). 36 LUTs, 0 DSP, exhaustively verified (all 243 bytes).
- **`ternary_gemv_packed` (`rtl/ternary_gemv_packed.sv`)** — the first integration:
  GEMV directly from dense base-3 packed rows (a `ternary_unpack5` array → `ternary_dot`),
  the shape of the real decode datapath (burst → unpack → MAC). Bit-exact, 0 DSP.

## Where it's going (planned `rtl/`)

| Module | Role | Phase |
|---|---|---|
| ✅ **`ternary_pe_array`** | `P` parallel dot lanes (bandwidth-matched, not FLOP-maxed) — P=4: 931 LUT, 0 DSP | ✅ done (sim) |
| ~~`sparse_skip`~~ → **`ternary_gemv_sparse`** | active-mask gather: fetch only active rows (Direction D) | ✅ done (sim) |
| ~~`weight_unpacker`~~ → **`ternary_unpack5`** | dense base-3 bytes → ternary codes (1.6 bits/weight) | ✅ done (sim) |
| `ddr3_stream` | MIG/LiteDRAM front-end + double-buffered tile feed | 1 |
| `requant` / `rmsnorm` / `rope` / `softmax` | the few real-multiply ops → the 90 DSP48 (Vitis HLS) | 1 |
| top `decode_core` | layer loop, KV in DDR3, UART/Eth token I/O | 1→2 |

**Design rule:** the array is sized to the **DDR3 sustained bandwidth roofline**
(~0.6–0.8 GB/s), not to peak FLOPs — more lanes than the memory can feed is wasted
area. The win is energy/byte and skipping bytes, not raw compute.

## Hardware budget (Arty A7-35T, `xc7a35t`)

20,800 LUTs · 41,600 FFs · **90 DSP48E1** · 1.8 Mb (~225 KB) BRAM · 256 MB DDR3L
(~0.6–0.8 GB/s sustained) · 100 Mb Eth + USB-UART. The ternary multiply path uses
**0 DSP**; the 90 DSPs are reserved for norm/RoPE/softmax/requant. Weights live in
DDR3 and stream (225 KB BRAM can't hold a model); sparsity shrinks that stream.

## Test strategy (TDD)

Every block ships with a cocotb testbench (`sim/tb_<name>.py`) that asserts
**bit-exact** equality against the NumPy golden — integer/ternary math, so exact,
not tolerance. `make -C sim` builds + runs each DUT via Verilator (cocotb's
per-DUT `Makefile.sim` flow) locally / on worker4, and CI runs the suite on every PR.
A block is "done" only with: RTL + passing bit-exact test + Vivado utilization
(DSP-free multiply path confirmed) + a logged measurement + a `BUILDLOG.md` entry.
