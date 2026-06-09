#!/usr/bin/env python3
"""LiteX Arty A7-35T SoC with the throughput harness (ternary_gemv_bench) as a peripheral.

  python soc/ternary_gemv_bench_arty.py          # gen only (validate integration)
  python soc/ternary_gemv_bench_arty.py --run    # full bitstream (Vivado, ~30 min)

The harness lets the CPU load resident weights, pulse `run`, and read back the exact
compute-cycle count — the on-silicon measurement behind the energy/token number.
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_gemv_bench_csr import TernaryGemvBench  # noqa: E402


def main():
    rtl = os.environ.get("RTL_DIR", "/srv/fpga/ternfpga/rtl")
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_bench")
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    soc.submodules.bench = TernaryGemvBench(soc.platform, rtl, K=8, NT_MAX=64, M_MAX=64, WDEPTH=4096)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="ternary_gemv_bench_arty", run=do_run)
    print(f"GEN_DONE run={do_run} out={out}")


if __name__ == "__main__":
    main()
