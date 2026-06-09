#!/usr/bin/env python3
"""LiteX Arty A7-35T SoC with attention_unit as a peripheral (on-fabric attention on silicon).

  python soc/attention_unit_arty.py          # gen only (validate integration)
  python soc/attention_unit_arty.py --run    # full bitstream (Vivado, ~30 min)
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_unit_csr import AttentionUnit  # noqa: E402


def main():
    rtl = os.environ.get("RTL_DIR", "/srv/fpga/ternfpga/rtl")
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_attn")
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    soc.submodules.attn = AttentionUnit(soc.platform, rtl, D=128, T_MAX=64, EXP_N=4096)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="attention_unit_arty", run=do_run)
    print(f"GEN_DONE run={do_run} out={out}")


if __name__ == "__main__":
    main()
