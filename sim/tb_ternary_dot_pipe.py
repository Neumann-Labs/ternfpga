"""cocotb test: pipelined ternary dot product, bit-exact vs NumPy.

Streams a new (a, w) every cycle, tracks expected results in FIFO order, and
checks each `valid_out` against the golden — so the 3-cycle pipeline latency is
handled by following `valid_out` rather than counting cycles. Also exercises
back-pressure-free streaming (one result/cycle once the pipe is full).
"""
import collections
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_dot_golden  # noqa: E402

K = int(os.environ.get("K", "8"))
N = int(os.environ.get("N_INPUTS", "800"))


def _to_signed(value, width: int = 32) -> int:
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def pipelined_dot(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.a_flat.value = 0
    dut.w_flat.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    rng = random.Random(2024)
    expected = collections.deque()
    got = 0

    # Drive one input/cycle for N cycles, then a few drain cycles; check each
    # valid_out against the FIFO of expected results.
    for n in range(N + 8):
        if n < N:
            a = [rng.randint(-128, 127) for _ in range(K)]
            w = [rng.choice([-1, 0, 1]) for _ in range(K)]
            dut.a_flat.value = pack_activations(a)
            dut.w_flat.value = pack_weights(w)
            dut.valid_in.value = 1
            expected.append(ternary_dot_golden(a, w))
        else:
            dut.valid_in.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        if int(dut.valid_out.value) == 1:
            d = _to_signed(dut.dot.value)
            e = expected.popleft()
            assert d == e, f"output #{got}: got={d} exp={e}"
            got += 1

    assert got == N, f"got {got} outputs, expected {N} (pipeline drained?)"
    assert not expected, f"{len(expected)} expected results never emitted"
    dut._log.info(f"PASS: {N} streamed pipelined ternary dots bit-exact, 1 result/cycle, valid-tracked")
