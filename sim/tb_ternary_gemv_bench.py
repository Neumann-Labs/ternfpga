"""cocotb test: ternary_gemv_bench — the on-silicon throughput harness.

Loads the activation + the weight tiles into the engine's resident BRAMs, pulses `run`,
and the replay FSM streams the weights at 1 tile/cycle. Checks y is bit-exact vs the dense
golden AND that the hardware `cycle_count` is ~1 cycle/tile (proving the engine sustains its
roofline from resident weights — the measurement the on-board energy/token rests on).
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402

K = 8


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def gemv_bench(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    for s in ("nt", "m_rows", "x_we", "x_waddr", "x_wdata",
              "wm_we", "wm_waddr", "wm_wdata", "run", "rd_addr"):
        getattr(dut, s).value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0xBE0)
    for nt, m in [(4, 8), (8, 16), (16, 32)]:
        KT = K * nt
        x = [rng.randint(-128, 127) for _ in range(KT)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(KT)] for _ in range(m)]
        dut.nt.value = nt
        dut.m_rows.value = m

        for t in range(nt):                              # load activation
            dut.x_we.value = 1
            dut.x_waddr.value = t
            dut.x_wdata.value = pack_activations(x[t * K:(t + 1) * K])
            await RisingEdge(dut.clk)
        dut.x_we.value = 0

        for mm in range(m):                              # load weight tiles row-major
            for t in range(nt):
                dut.wm_we.value = 1
                dut.wm_waddr.value = mm * nt + t
                dut.wm_wdata.value = pack_weights(W[mm][t * K:(t + 1) * K])
                await RisingEdge(dut.clk)
        dut.wm_we.value = 0
        await RisingEdge(dut.clk)

        dut.run.value = 1                                # measured run
        await RisingEdge(dut.clk)
        dut.run.value = 0
        for _ in range(m * nt + 64):
            await RisingEdge(dut.clk)
            if int(dut.done.value) == 1:
                break
        else:
            assert False, f"nt={nt} m={m}: done never asserted"

        exp = ternary_gemv_golden(W, x)
        for mm in range(m):
            dut.rd_addr.value = mm
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
            got = _to_signed(dut.rd_data.value)
            assert got == exp[mm], f"nt={nt} m={m} row {mm}: got={got} exp={exp[mm]}"

        cc = int(dut.cycle_count.value)
        total = nt * m
        assert total <= cc <= total + 24, f"cycle_count {cc} not ~1 cyc/tile (total {total})"
        dut._log.info(f"nt={nt} m={m}: y bit-exact; cycle_count={cc} for {total} tiles "
                      f"({cc / total:.2f} cyc/tile)")

    dut._log.info("PASS: ternary_gemv_bench - 1 tile/cycle from resident weights, cycle_count measured")
