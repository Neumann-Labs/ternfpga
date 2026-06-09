# Measured DDR3 read roofline — the Risk-1 verdict

Phase-3 #28. A hardware `LiteDRAMDMAReader` on a native crossbar port streams a contiguous
region from DDR3 while a hardware counter times it (`soc/ternary_dma_bw.py`). This is the TRUE
sustained read bandwidth — not the BIOS Memspeed (the CPU's slow wishbone memcpy, 37–49 MiB/s).

## Measured on silicon (firmware over UART, 100 MHz)
```
=== ternfpga DDR3 read-bandwidth roofline ===
native word = 16 bytes; reading 2000000 words = 30 MiB
words=2000000 cycles=2248263 chksum=0x7a84c76d
DDR3_READ_BW  bytes=32000000 cycles=2248263  bw=1423 MB/s
ROOFLINE  model=175 MB/tok  tok_s_ceiling=8.13  Jtok_floor~=60 mJ (est 489 mW SoC)
```
- **Sustained DDR3 read = 1423 MB/s** = **0.89 word/cycle** on a 128-bit native port =
  **89% of the 1.6 GB/s peak** (the 11% gap is refresh + row activation). Non-zero checksum
  proves real bytes moved. _4× higher than the original ~0.35 GB/s grounding-doc guess._

## What it means (the roofline)

Batch-1 LLM decode is **bandwidth-bound** — every weight is streamed once per token — so this
single number caps full-model throughput and floors energy:

| | value |
|---|---:|
| Sustained DDR3 read | **1.42 GB/s** |
| K=8 engine demand (1 tile/cyc × 2 B × 100 MHz) | 200 MB/s → **compute-bound (14% of DDR3)** |
| K to saturate DDR3 | **K ≈ 56** (≈ 7 tiles/cycle equiv) |
| 0.7B ternary model (2-bit packed) | ~175 MB/token |
| **tok/s ceiling** (1423 / 175) | **8.1 tok/s** |
| **J/token floor** (489 mW est. ÷ 8.1) | **~60 mJ/token** |

## Risk-1 verdict: the energy thesis SURVIVES single-channel DDR3 ✅

The fear (project plan, Risk 1) was that a $130 board's single 16-bit DDR3 channel would cap
throughput so low the energy advantage evaporates. It does not:

- **8.1 tok/s** is usable and the same order as TeLLMe v2's 25 tok/s on a Kria KV260 — which has
  *faster DDR4 (~17 GB/s aggregate)* and costs more. For one narrow DDR3 channel on a $130 Artix,
  1.42 GB/s is healthy.
- The **~60 mJ/token floor** is ~20–60× under the RTX 3060's full-model energy (3.67 J/tok on
  BitNet-2B; even scaling that GPU number down to 0.7B leaves a ~20× gap). The bandwidth roofline
  does **not** sink the headline.
- The engine is **compute-bound at K=8** (confirms the earlier finding): the path to using the
  full channel is to **widen K**, not to find more bandwidth.

## Honesty / scope

- This is a **floor**, not the delivered number — it assumes the engine + glue keep the channel
  saturated with zero decode-loop overhead. The **real measured tok/s + J/token** comes from the
  actual decode loop (#31/#32) and will be higher (VexRiscv softmax/RoPE/RMSNorm glue is slow).
- The GPU comparison mixes model sizes (3060 measured on 2B, this floor on 0.7B) — the
  apples-to-apples number is the decode-loop measurement on the same 0.7B model.
- Power is the **Vivado estimate** (489 mW SoC), clearly labeled; live metering deferred.
- The measurement is a counter over a real DMA (no datapath correctness to simulate); the
  DMA→engine *correctness* is simulated when the two integrate (#30).

_Reproduce:_ `python soc/ternary_dma_bw_arty.py --run`; flash + `main_dmabw.c` per `soc/README.md`.
