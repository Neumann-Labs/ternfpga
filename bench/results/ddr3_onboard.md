# DDR3 working on the board — LiteX/LiteDRAM bring-up

The hard prerequisite for the on-board inference datapath: **256 MB DDR3 calibrated
and verified on the physical Arty A7-35T**, via a LiteX SoC (VexRiscv + LiteDRAM).
2026-06-08. Built with LiteX 2026.4, flashed with openFPGALoader, BIOS read over
`/dev/ttyUSB1` @ 115200.

## Result — LiteX BIOS over UART
```
BIOS CRC passed (a9233342)   LiteX git sha1: d100f9d80
CPU:       VexRiscv @ 100MHz
BUS:       wishbone 32-bit data/32-bit addr
SDRAM:     256.0MiB 16-bit @ 800MT/s (CL-7 CWL-5)
MAIN RAM:  256.0MiB

Initializing SDRAM @0x40000000...
Read leveling:
  best: m0, b03 delays: 20+-06
  best: m1, b03 delays: 20+-06        <-- per-byte-lane delays CALIBRATED
Memtest at 0x40000000 (2.0MiB)... Memtest OK     <-- DDR3 VERIFIED, no bit errors
Memspeed:  Write 36.9MiB/s   Read 48.7MiB/s
litex>                                            <-- BIOS console, SoC alive
```

**Read leveling calibrated and Memtest OK** — the MT41K128M16 DDR3 is functional on
this board at 800 MT/s. This is the single hardest step of Phase 1 and it passes:
the memory the ternary engine will stream weights from is proven on silicon.

The BIOS `Memspeed` (37–49 MiB/s) is the **CPU's memcpy** rate, *not* the LiteDRAM
roofline — the native LiteDRAM port delivers far more (the grounding doc's ~0.6–0.8
GB/s sustained is the figure the engine is sized against). A hardware DMA, not the
CPU, will feed the engine.

## Reproduce
`syn/litex_arty.sh` (deps + build), then flash + read UART (see the script footer).

## Next
Integrate `ternary_tile` into the SoC as a CSR/DMA peripheral: stage a weight tile
in DDR3, stream the dense base-3 byte burst into the engine, read back `y` — which
finally yields a **measured on-board tokens/sec** and (with `report_power` + a SAIF)
a **measured FPGA J/token** to drop into the head-to-head (`gpu_baseline.md`).
