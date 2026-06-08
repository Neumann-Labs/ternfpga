"""cocotb testbench: ternary_gemv must be bit-exact vs the NumPy golden (W @ x).

Streams a random ternary weight matrix one row per cycle, then reads the result
vector back through the narrow rd_addr/rd_data port. This is the matmul
primitive the whole engine is built on.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import ternary_gemv_golden, pack_activations, pack_weights  # noqa: E402

K = int(os.environ.get("K", "8"))
M = int(os.environ.get("M", "16"))
N_TRIALS = int(os.environ.get("N_TRIALS", "60"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def random_gemv(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset.
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.w_row_valid.value = 0
    dut.w_row.value = 0
    dut.x_flat.value = 0
    dut.rd_addr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0xA11CE)
    for trial in range(N_TRIALS):
        x = [rng.randint(-128, 127) for _ in range(K)]
        # Trial 0: directed all-zero / all-+1 / all--1 rows; otherwise random.
        W = []
        for m in range(M):
            if trial == 0 and m < 3:
                W.append([[0, 1, -1][m]] * K)
            else:
                W.append([rng.choice([-1, 0, 1]) for _ in range(K)])

        # start: latch x.
        dut.x_flat.value = pack_activations(x)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        # stream M rows, one per cycle.
        for m in range(M):
            dut.w_row.value = pack_weights(W[m])
            dut.w_row_valid.value = 1
            await RisingEdge(dut.clk)
        dut.w_row_valid.value = 0
        await RisingEdge(dut.clk)   # let the final write settle
        await Timer(1, units="ns")

        assert int(dut.done.value) == 1, f"trial {trial}: done not asserted"

        # read back the result vector through the address port.
        exp = ternary_gemv_golden(W, x)
        for m in range(M):
            dut.rd_addr.value = m
            await Timer(1, units="ns")
            got = _to_signed(dut.rd_data.value)
            assert got == exp[m], f"trial {trial} row {m}: got={got} exp={exp[m]} (x={x}, W[m]={W[m]})"

    dut._log.info(f"PASS: {N_TRIALS} random K={K} M={M} ternary GEMVs bit-exact vs numpy")
