"""cocotb test: column-sparse (activation-sparse) down_proj via gather, bit-exact.

down_proj is y[h] = sum_f Wd[h,f]*hq[f] with the activation hq ~60% zero per token
(measured, bench/results/activation_sparsity.md). The zero terms contribute nothing, so
we COMPACT hq to its nonzero entries and GATHER the matching Wd columns, then run the
*existing* dense streaming GEMV on the shorter vector. Result: y is bit-exact vs the dense
golden while only `active/F` of the weight bytes are ever fetched — a per-token, unstructured
fetch reduction a GPU's dense (or 2:4) array can't do. The engine is UNCHANGED; the gather is
a feed/DMA concern (the DMA fetches only the gathered columns).

Reports the weight-byte savings across densities; the hardware index-compaction + DMA gather
is task #24.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_gemv_golden  # noqa: E402

K = 16


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


async def run_gemv(dut, x_q, nt, m, W):
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


def gather(hq, Wd):
    """Compact nonzero hq + gather the matching Wd columns; pad to a multiple of K with zeros."""
    nz = [f for f, v in enumerate(hq) if v != 0]
    pad = (-len(nz)) % K
    idx = nz + [0] * pad                      # padded indices (value 0 at pads -> 0 contribution)
    hq_c = [hq[f] for f in nz] + [0] * pad
    Wd_c = [[Wd[m][f] for f in nz] + [0] * pad for m in range(len(Wd))]
    return hq_c, Wd_c, len(nz)


@cocotb.test()
async def gemv_gather(dut):
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

    rng = random.Random(0x6A7E)
    F, H = 256, 24                              # down_proj: F contraction (sparse), H outputs
    rows = []
    for density in (1.00, 0.60, 0.402, 0.15):  # 0.402 = the measured BitNet active fraction
        hq = []
        for _ in range(F):
            if rng.random() < density:
                v = rng.randint(-100, 100) or 7
            else:
                v = 0
            hq.append(v)
        Wd = [[rng.choice([-1, 0, 1]) for _ in range(F)] for _ in range(H)]
        exp = ternary_gemv_golden(Wd, hq)       # dense reference

        hq_c, Wd_c, nz = gather(hq, Wd)
        nt = len(hq_c) // K
        got = await run_gemv(dut, hq_c, nt, H, Wd_c)
        assert got == exp, f"density {density}: gathered GEMV != dense golden"

        dense_tiles = ((F + K - 1) // K) * H
        gathered_tiles = nt * H
        saved = 100.0 * (1 - gathered_tiles / dense_tiles)
        rows.append((density, nz, F, saved))
        dut._log.info(f"density {density:.3f}: active {nz}/{F}, weight-tiles "
                      f"{gathered_tiles}/{dense_tiles}, saved {saved:.1f}% - bit-exact")

    dut._log.info("PASS: down_proj column-sparse gather bit-exact vs dense; byte-fetch scales "
                  "with activation density (engine unchanged, gather in the feed)")
