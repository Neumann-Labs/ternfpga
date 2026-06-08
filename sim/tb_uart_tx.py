"""cocotb test: 8N1 UART transmitter, bit-exact byte round-trip.

Decodes the serial line by sampling at bit centers and checks each transmitted
byte matches what was sent. Uses a small CLKS_PER_BIT (override -GCLKS_PER_BIT=8)
so the test is fast.
"""
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

CPB = int(os.environ.get("CLKS_PER_BIT", "8"))


async def uart_decode(dut):
    """Wait for a start bit, sample 8 data bits LSB-first at their centers."""
    await FallingEdge(dut.tx)                       # start bit begins
    await ClockCycles(dut.clk, CPB + CPB // 2)      # -> center of data bit 0
    byte = 0
    for i in range(8):
        byte |= (int(dut.tx.value) & 1) << i
        if i < 7:
            await ClockCycles(dut.clk, CPB)
    await ClockCycles(dut.clk, CPB)                 # -> stop bit
    assert int(dut.tx.value) == 1, "stop bit must be high"
    return byte


@cocotb.test()
async def send_bytes(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.data.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = random.Random(7)
    payload = [0x00, 0xFF, 0xA5, 0x5A, ord("T"), ord("N")] + [rng.randint(0, 255) for _ in range(8)]
    for b in payload:
        decoder = cocotb.start_soon(uart_decode(dut))
        await RisingEdge(dut.clk)
        dut.data.value = b
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        got = await decoder
        assert got == b, f"sent {b:#04x}, decoded {got:#04x}"
        while int(dut.busy.value) == 1:
            await RisingEdge(dut.clk)

    dut._log.info(f"PASS: {len(payload)} UART bytes transmitted + decoded bit-exact (8N1)")
