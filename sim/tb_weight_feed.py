"""cocotb test: weight_feed bridges a dense base-3 byte stream -> w_row stream.

Feeds N rows' worth of packed bytes (1/cycle, via export_weights.pack_row_trits5)
and checks each emitted w_row matches the 2-bit codes for that row — proving the
memory-burst -> unpack -> row-stream path the on-board datapath needs.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_weights  # noqa: E402
from export_weights import pack_row_trits5  # noqa: E402

K = int(os.environ.get("K", "10"))
N = int(os.environ.get("N_ROWS", "50"))


@cocotb.test()
async def feed(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.byte_in.value = 0
    dut.byte_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(0xFEED)
    rows = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(N)]
    expected = [pack_weights(W) for W in rows]
    stream = b"".join(pack_row_trits5(W) for W in rows)

    collected = []

    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
            if int(dut.row_valid.value) == 1:
                collected.append(int(dut.w_row.value))

    cocotb.start_soon(monitor())

    for byte in stream:
        dut.byte_in.value = int(byte)
        dut.byte_valid.value = 1
        await RisingEdge(dut.clk)
    dut.byte_valid.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)

    assert len(collected) == N, f"got {len(collected)} rows, expected {N}"
    for i, (got, exp) in enumerate(zip(collected, expected)):
        assert got == exp, f"row {i}: got={got:#x} exp={exp:#x}"
    dut._log.info(f"PASS: weight_feed bridged {N} rows (base-3 byte stream -> w_row codes), bit-exact")
