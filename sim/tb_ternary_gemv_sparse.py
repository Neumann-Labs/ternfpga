"""cocotb testbench: activation-sparse ternary GEMV.

Verifies (1) bit-exactness vs the NumPy golden (active rows = dot, inactive = 0),
and (2) that the engine fetches EXACTLY the active rows — i.e. it does not touch
weight memory for skipped neurons. Logs the measured weight-byte saving across
sparsity levels, which is Direction D's whole thesis in miniature.

A coroutine models a synchronous-read weight ROM (data valid 1 cycle after ren).
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import ternary_gemv_sparse_golden, pack_activations, pack_weights  # noqa: E402

K = int(os.environ.get("K", "8"))
M = int(os.environ.get("M", "16"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


async def weight_memory(dut, mem):
    """Synchronous-read weight ROM: mem_rdata reflects the last addr latched on mem_ren."""
    addr_q = 0
    dut.mem_rdata.value = pack_weights(mem["W"][0])
    while True:
        await RisingEdge(dut.clk)
        if int(dut.mem_ren.value):
            addr_q = int(dut.mem_addr.value)
        dut.mem_rdata.value = pack_weights(mem["W"][addr_q])


async def _wait_done(dut, max_cycles):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    raise AssertionError("timeout waiting for done")


@cocotb.test()
async def sparse_gemv(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    mem = {"W": [[0] * K for _ in range(M)]}
    cocotb.start_soon(weight_memory(dut, mem))

    # reset
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x_flat.value = 0
    dut.active_mask.value = 0
    dut.rd_addr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0x5A25E)
    dense_bytes = M * K * 2 // 8  # ternary = 2 bits/weight

    levels = [("dense", M), ("75%", (3 * M) // 4), ("50%", M // 2),
              ("25%", M // 4), ("sparse1", 1), ("empty", 0)]
    for label, n_active in levels:
        x = [rng.randint(-128, 127) for _ in range(K)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(M)]
        active = sorted(rng.sample(range(M), n_active))
        mask = [1 if m in active else 0 for m in range(M)]
        mem["W"] = W

        dut.x_flat.value = pack_activations(x)
        dut.active_mask.value = sum(mask[m] << m for m in range(M))
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        await _wait_done(dut, 4 * M + 32)
        await Timer(1, units="ns")

        # correctness: active rows = dot, inactive rows = 0
        exp = ternary_gemv_sparse_golden(W, x, mask)
        for m in range(M):
            dut.rd_addr.value = m
            await Timer(1, units="ns")
            got = _to_signed(dut.rd_data.value)
            assert got == exp[m], f"{label} row {m}: got={got} exp={exp[m]}"

        # the lever: fetched exactly the active rows, nothing more
        fetched = int(dut.rows_fetched.value)
        assert fetched == n_active, f"{label}: rows_fetched={fetched} != active={n_active}"

        fetched_bytes = fetched * K * 2 // 8
        saved = 100.0 * (1.0 - fetched / M) if M else 0.0
        dut._log.info(
            f"{label:8s} active={n_active:2d}/{M}  rows_fetched={fetched:2d}  "
            f"weight_bytes={fetched_bytes:3d}/{dense_bytes}  saved={saved:5.1f}%"
        )

    dut._log.info("PASS: sparse GEMV bit-exact and fetch-count == active rows across all densities")
