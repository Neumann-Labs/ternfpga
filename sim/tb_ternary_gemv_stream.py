"""cocotb test: BRAM-centric streaming ternary GEMV, bit-exact over real widths.

Loads the activation into the engine's BRAM, streams weight tiles row-major, and
checks y = W·x (W is M×KT, x is KT, KT = K·nt) against the NumPy golden — proving
the BRAM-resident, sequential-address, pipelined datapath that replaces the
register-resident flat-mux design (fit sweep: that one failed timing at width).
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402

K = int(os.environ.get("K", "16"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


async def run_one(dut, rng, nt: int, m: int):
    KT = K * nt
    x = [rng.randint(-128, 127) for _ in range(KT)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(KT)] for _ in range(m)]

    dut.start.value = 0
    dut.w_valid.value = 0
    dut.x_we.value = 0
    dut.nt.value = nt
    dut.m_rows.value = m
    await RisingEdge(dut.clk)

    # load nt activation words (K int8 each) into the engine BRAM
    for t in range(nt):
        dut.x_we.value = 1
        dut.x_waddr.value = t
        dut.x_wdata.value = pack_activations(x[t * K:(t + 1) * K])
        await RisingEdge(dut.clk)
    dut.x_we.value = 0

    # start: latch dims, reset counters
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # stream weight tiles row-major (row 0 tiles 0..nt-1, row 1 ...)
    for mm in range(m):
        for t in range(nt):
            dut.w_valid.value = 1
            dut.w_tile.value = pack_weights(W[mm][t * K:(t + 1) * K])
            await RisingEdge(dut.clk)
    dut.w_valid.value = 0

    for _ in range(m * nt + 64):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, f"done never asserted (nt={nt} m={m})"

    exp = ternary_gemv_golden(W, x)
    for mm in range(m):
        dut.rd_addr.value = mm
        await RisingEdge(dut.clk)          # registered read: data valid after this edge
        await Timer(1, units="ns")
        got = _to_signed(dut.rd_data.value)
        assert got == exp[mm], f"nt={nt} m={m} row {mm}: got={got} exp={exp[mm]}"


@cocotb.test()
async def gemv_stream(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    for sig in ("start", "w_valid", "x_we", "w_tile", "x_wdata", "x_waddr", "rd_addr"):
        getattr(dut, sig).value = 0
    dut.nt.value = 1
    dut.m_rows.value = 1
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0xB17E7)
    configs = [(1, 4), (4, 8), (8, 16), (16, 32), (13, 7), (32, 5)]  # (nt, m); odd + single-tile
    for nt, m in configs:
        await run_one(dut, rng, nt, m)

    dut._log.info(f"PASS: ternary_gemv_stream bit-exact over {len(configs)} configs "
                  f"(K={K}, BRAM-centric, pipelined, 0 DSP)")
