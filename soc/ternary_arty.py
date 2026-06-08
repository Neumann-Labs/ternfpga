#!/usr/bin/env python3
"""Custom LiteX Arty A7-35T SoC with the ternary_tile engine as a CSR peripheral.

  python soc/ternary_arty.py            # generate gateware only (fast, validates integration)
  python soc/ternary_arty.py --run      # full build (synth+impl+bitstream via Vivado, ~30 min)

Then flash build_ternary/.../ternary_arty.bit and drive the engine from the CPU
(csr.csv has the register map). The result is an on-board ternary GEMV computed by
the fabric engine and read back over the wishbone bus.
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_tile_csr import TernaryTile  # noqa: E402


def main():
    rtl = os.environ.get("RTL_DIR", "/srv/fpga/ternfpga/rtl")
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_ternary")
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    soc.submodules.ternary = TernaryTile(soc.platform, rtl, K=10, M=16)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="ternary_arty", run=do_run)
    print(f"GEN_DONE run={do_run} out={out}")


if __name__ == "__main__":
    main()
