"""cocotb test: row-streamed GEMV using the pipelined dot lane, bit-exact.

Streams M weight rows back-to-back (1/cycle), waits for `done` (the engine
absorbs the 3-cycle pipeline latency via valid-tracking), then checks y == W·x.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402

K = int(os.environ.get("K", "8"))
M = int(os.environ.get("M", "16"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def gemv_pipe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
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

    rng = random.Random(0xC0FFEE)
    for t in range(40):
        x = [rng.randint(-128, 127) for _ in range(K)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(M)]

        dut.x_flat.value = pack_activations(x)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        for m in range(M):
            dut.w_row.value = pack_weights(W[m])
            dut.w_row_valid.value = 1
            await RisingEdge(dut.clk)
        dut.w_row_valid.value = 0

        for _ in range(M + 16):
            await RisingEdge(dut.clk)
            if int(dut.done.value) == 1:
                break
        else:
            assert False, f"trial {t}: done never asserted"
        await Timer(1, units="ns")

        exp = ternary_gemv_golden(W, x)
        for m in range(M):
            dut.rd_addr.value = m
            await Timer(1, units="ns")
            got = _to_signed(dut.rd_data.value)
            assert got == exp[m], f"trial {t} row {m}: got={got} exp={exp[m]}"

    dut._log.info(f"PASS: 40 pipelined-lane K={K} M={M} GEMVs bit-exact (streaming, valid-tracked, 0 DSP)")
