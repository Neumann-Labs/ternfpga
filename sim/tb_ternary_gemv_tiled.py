"""cocotb test: tiled ternary GEMV over a full-width row (KT = K*NT), bit-exact.

Streams M*NT weight tiles row-major against a stationary KT-wide activation and
checks y = W·x for the full row width — proving the K-tiling accumulation that
lets the fixed lane handle real layer dims.
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
NT = int(os.environ.get("NT", "4"))
M = int(os.environ.get("M", "16"))
KT = K * NT


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def gemv_tiled(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.w_tile_valid.value = 0
    dut.w_tile.value = 0
    dut.x_flat.value = 0
    dut.rd_addr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0x71117)
    for t in range(25):
        x = [rng.randint(-128, 127) for _ in range(KT)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(KT)] for _ in range(M)]

        dut.x_flat.value = pack_activations(x)
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        for m in range(M):
            for tt in range(NT):
                dut.w_tile.value = pack_weights(W[m][tt * K:(tt + 1) * K])
                dut.w_tile_valid.value = 1
                await RisingEdge(dut.clk)
        dut.w_tile_valid.value = 0

        for _ in range(M * NT + 24):
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

    dut._log.info(f"PASS: 25 tiled GEMVs bit-exact (K={K} NT={NT} KT={KT} M={M}, 0 DSP)")
