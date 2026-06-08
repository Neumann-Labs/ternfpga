"""cocotb test: run a REAL BitNet ternary weight tile through ternary_gemv.

Loads an exported tile (`models/data/real_tile.npz`, produced by
`models/extract_bitnet_layer.py`) and checks the engine is bit-exact vs the
NumPy golden on actual trained ternary weights — proving the model→RTL path,
not just random stimulus.
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402
from export_weights import load_tile  # noqa: E402

K = int(os.environ.get("K", "8"))
M = int(os.environ.get("M", "16"))
TILE = os.environ.get("TILE_NPZ",
                       os.path.join(os.path.dirname(__file__), "..", "models", "data", "real_tile.npz"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def real_layer_tile(dut):
    W, x = load_tile(TILE)            # W: MxK ternary int, x: K int8
    assert W.shape == (M, K), f"tile {W.shape} != ({M},{K}); set M/K to match the export"

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

    dut.x_flat.value = pack_activations([int(v) for v in x])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for m in range(M):
        dut.w_row.value = pack_weights([int(v) for v in W[m]])
        dut.w_row_valid.value = 1
        await RisingEdge(dut.clk)
    dut.w_row_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")

    assert int(dut.done.value) == 1, "done not asserted"
    exp = ternary_gemv_golden(W, x)
    for m in range(M):
        dut.rd_addr.value = m
        await Timer(1, units="ns")
        got = _to_signed(dut.rd_data.value)
        assert got == exp[m], f"row {m}: got={got} exp={exp[m]} (W[m]={list(W[m])})"

    nz = int((W != 0).sum())
    tot = M * K
    dut._log.info(f"PASS: REAL BitNet ternary tile {M}x{K} bit-exact vs numpy "
                  f"(tile density={nz}/{tot}={100.0*nz/tot:.0f}% nonzero)")
