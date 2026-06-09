#!/usr/bin/env python3
"""LiteX Arty A7-35T SoC with the BRAM-centric streaming ternary GEMV as a CSR peripheral.

  python soc/ternary_gemv_arty.py          # generate gateware only (fast; validates integration)
  python soc/ternary_gemv_arty.py --run    # full build (synth+impl+bitstream via Vivado, ~30 min)

Then flash build_gemv/.../ternary_gemv_arty.bit and drive the engine from the CPU
(csr.csv has the register map). This is the scalable, real-width GEMV running on the
fabric — the engine the FFN block (gate/up/down) uses — read back over the bus.
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_gemv_csr import TernaryGemvStream  # noqa: E402


def main():
    rtl = os.environ.get("RTL_DIR", "/srv/fpga/ternfpga/rtl")
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_gemv")
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    soc.submodules.gemv = TernaryGemvStream(soc.platform, rtl, K=8, NT_MAX=64, M_MAX=64)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="ternary_gemv_arty", run=do_run)
    print(f"GEN_DONE run={do_run} out={out}")


if __name__ == "__main__":
    main()
