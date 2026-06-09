#!/usr/bin/env python3
"""LiteX Arty A7-35T SoC with BOTH the ternary engine and the FFN-glue unit co-resident — the
integrated capstone (two custom accelerators cooperating on one board).

  python soc/coresident_arty.py          # gen only (validate integration + BRAM fit)
  python soc/coresident_arty.py --run    # full bitstream (Vivado, ~30 min)

The firmware (main_coresident.c) runs a small FFN: the engine computes the gate/up ternary GEMVs,
its int32 outputs feed the ffn_glue unit (relu^2*up*w + int8 requant), and h_q is read back —
bit-exact, end-to-end, on real silicon. Engine weights are streamed (minimal BRAM); ffn_glue holds
the full real width (F_MAX=6912, 20 BRAM) to prove the real-width unit co-resides with the engine.
"""
import os
import sys

from litex_boards.targets.digilent_arty import BaseSoC
from litex.soc.integration.builder import Builder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_gemv_csr import TernaryGemvStream  # noqa: E402
from ffn_glue_csr import FfnGlueUnit            # noqa: E402


def main():
    rtl = os.environ.get("RTL_DIR", "/srv/fpga/ternfpga/rtl")
    out = os.environ.get("OUT_DIR", "/srv/fpga/litex-build/build_cores")
    fmax = int(os.environ.get("FFN_FMAX", "6912"))
    do_run = "--run" in sys.argv

    soc = BaseSoC(variant="a7-35", sys_clk_freq=100e6, integrated_rom_size=0x20000)
    soc.submodules.eng = TernaryGemvStream(soc.platform, rtl, K=8, NT_MAX=64, M_MAX=64)
    soc.submodules.ffng = FfnGlueUnit(soc.platform, rtl, F_MAX=fmax)

    builder = Builder(soc, output_dir=out, csr_csv=os.path.join(out, "csr.csv"))
    builder.build(build_name="coresident_arty", run=do_run)
    print(f"GEN_DONE run={do_run} fmax={fmax} out={out}")


if __name__ == "__main__":
    main()
