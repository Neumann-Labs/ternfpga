"""cocotb test: the full BitNet FFN block, end-to-end through the real streaming GEMV.

The FFN block is host-split: the FPGA runs the three ternary matmuls (`ternary_gemv_stream`),
the host does the (now integer-only, per ffn_glue_ref) glue between them. This drives the actual
RTL GEMV three times — gate, up, down — with the host glue in Python in between, and checks every
integer stage bit-exact against the validated `ffn_ref` golden:

    gate_int = stream_gemv(Wg, x_q)            # FPGA
    up_int   = stream_gemv(Wu, x_q)            # FPGA  (x reused)
    h_q      = glue_hq(gate_int, up_int, w)    # host: relu^2 . up . w, absmax-requant (ints only)
    down_int = stream_gemv(Wd, h_q)            # FPGA

Toplevel is `ternary_gemv_stream` — there is no separate FFN RTL; the block *is* the GEMV used 3×
plus host glue. Proves the host-split FFN datapath on the real hardware engine.
"""
import os
import random
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402
from ffn_ref import act_quant_int8, ffn_block as ffn_block_ref  # noqa: E402
from ffn_glue_ref import glue_hq  # noqa: E402

K = 16


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


async def run_gemv(dut, x_q, nt, m, W):
    """Drive one GEMV through the RTL: load activation, stream weight tiles, read y. x_q: nt*K int8."""
    dut.start.value = 0
    dut.w_valid.value = 0
    dut.x_we.value = 0
    dut.nt.value = nt
    dut.m_rows.value = m
    await RisingEdge(dut.clk)
    for t in range(nt):
        dut.x_we.value = 1
        dut.x_waddr.value = t
        dut.x_wdata.value = pack_activations(x_q[t * K:(t + 1) * K])
        await RisingEdge(dut.clk)
    dut.x_we.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
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
    out = []
    for mm in range(m):
        dut.rd_addr.value = mm
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        out.append(_to_signed(dut.rd_data.value))
    return out


@cocotb.test()
async def ffn_through_rtl(dut):
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

    rng = random.Random(0xFedFa)
    nprng = np.random.default_rng(7)
    H, F = 32, 64                                  # hidden, ff (multiples of K=16)

    for trial in range(4):
        x = nprng.standard_normal((1, H)) * (1.0 + trial)        # one token
        Wg = [[rng.choice([-1, 0, 1]) for _ in range(H)] for _ in range(F)]
        Wu = [[rng.choice([-1, 0, 1]) for _ in range(H)] for _ in range(F)]
        Wd = [[rng.choice([-1, 0, 1]) for _ in range(F)] for _ in range(H)]
        norm_w = nprng.uniform(0.5, 1.5, size=F)
        # scales don't affect the integer intermediates (they cancel) — use 1.0
        _, inter = ffn_block_ref(x, Wg, 1.0, Wu, 1.0, Wd, 1.0, norm_w, 1e-5)

        x_q = act_quant_int8(x)[0][0].tolist()                   # H int8

        gate_rtl = await run_gemv(dut, x_q, H // K, F, Wg)
        assert gate_rtl == inter["gate_int"][0].tolist(), f"trial {trial}: gate_int mismatch"
        up_rtl = await run_gemv(dut, x_q, H // K, F, Wu)
        assert up_rtl == inter["up_int"][0].tolist(), f"trial {trial}: up_int mismatch"

        # host glue (integer-only): h_q from the RTL gate/up outputs
        hq = glue_hq(np.array([gate_rtl]), np.array([up_rtl]), norm_w)[0][0].tolist()
        gmatch = float(np.mean(np.array(hq) == inter["down_xq"][0])) * 100
        assert gmatch >= 99.0, f"trial {trial}: glue h_q only {gmatch:.1f}% vs ffn_ref"

        down_rtl = await run_gemv(dut, hq, F // K, H, Wd)
        assert down_rtl == ternary_gemv_golden(Wd, hq), f"trial {trial}: down_int mismatch"

    dut._log.info("PASS: full FFN block end-to-end through ternary_gemv_stream RTL "
                  "(gate/up/down bit-exact, host glue integer-only) over 4 trials")
