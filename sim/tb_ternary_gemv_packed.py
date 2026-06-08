"""cocotb test: matrix-vector over DENSE base-3 packed weight rows.

Drives weight rows in the on-DDR3 dense layout (5 ternary weights/byte via
`export_weights.pack_row_trits5`); the RTL unpacks them through ternary_unpack5
and feeds the multiply-free dot. Checks the full GEMV bit-exact vs the NumPy
golden — proving the packed-weight → unpack → MAC datapath end to end.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, ternary_gemv_golden  # noqa: E402
from export_weights import pack_row_trits5  # noqa: E402

K = int(os.environ.get("K", "10"))
M = int(os.environ.get("M", "16"))
TRIALS = int(os.environ.get("TRIALS", "40"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def gemv_packed(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.w_row_valid.value = 0
    dut.w_row_packed.value = 0
    dut.x_flat.value = 0
    dut.rd_addr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(99)
    for t in range(TRIALS):
        x = [rng.randint(-128, 127) for _ in range(K)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(M)]

        dut.x_flat.value = pack_activations(x)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        for m in range(M):
            packed = pack_row_trits5(W[m])               # BPR dense base-3 bytes
            dut.w_row_packed.value = int.from_bytes(packed, "little")
            dut.w_row_valid.value = 1
            await RisingEdge(dut.clk)
        dut.w_row_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")

        assert int(dut.done.value) == 1, f"trial {t}: done not set"
        exp = ternary_gemv_golden(W, x)
        for m in range(M):
            dut.rd_addr.value = m
            await Timer(1, units="ns")
            got = _to_signed(dut.rd_data.value)
            assert got == exp[m], f"trial {t} row {m}: got={got} exp={exp[m]}"

    dut._log.info(f"PASS: {TRIALS} dense-packed K={K} M={M} GEMVs bit-exact "
                  "(weights via base-3 unpack -> multiply-free dot, 0 DSP)")
