#!/usr/bin/env python3
"""LiteX Arty A7-35T SoC with the DDR3 read-bandwidth instrument (ternary_dma_bw).

  python soc/ternary_dma_bw_arty.py          # gen only (validate integration)
  python soc/ternary_dma_bw_arty.py --run    # full bitstream (Vivado, ~30 min)

Adds one native LiteDRAM read port driven by a hardware DMA + cycle counter, so the CPU
can measure the TRUE sustained DDR3 read bandwidth (the roofline behind the tok/s ceiling).
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_dma_bw import TernaryDmaBw  # noqa: E402


def main():
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_dmabw")
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    port = soc.sdram.crossbar.get_port(mode="read")          # native read master
    soc.submodules.dmabw = TernaryDmaBw(port)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="ternary_dma_bw_arty", run=do_run)
    print(f"GEN_DONE run={do_run} out={out}")


if __name__ == "__main__":
    main()
