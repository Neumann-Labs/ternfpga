"""cocotb integration test for arty_top: decode its UART, verify y == 2*c.

Runs the full on-board top (ternary engine + message FSM + UART) with a small
CLKS_PER_BIT, samples uart_tx_o at bit centers to recover the ASCII stream, and
checks the engine's reported result matches the golden — the same check the
on-silicon verifier does, but in sim (so RTL bugs surface here, not on the board).
"""
import os
import re

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

CPB = int(os.environ.get("CLKS_PER_BIT", "8"))


async def uart_rx(dut, sink):
    """Continuously decode bytes off uart_tx_o into `sink`."""
    while True:
        await FallingEdge(dut.uart_tx_o)                 # start bit
        await ClockCycles(dut.CLK100MHZ, CPB + CPB // 2)  # center of data bit 0
        b = 0
        for i in range(8):
            b |= (int(dut.uart_tx_o.value) & 1) << i
            if i < 7:
                await ClockCycles(dut.CLK100MHZ, CPB)
        sink.append(b)
        await ClockCycles(dut.CLK100MHZ, CPB)             # stop bit


@cocotb.test()
async def onboard_compute(dut):
    cocotb.start_soon(Clock(dut.CLK100MHZ, 10, units="ns").start())
    sink = bytearray()
    cocotb.start_soon(uart_rx(dut, sink))
    await ClockCycles(dut.CLK100MHZ, 14000)               # let several messages stream

    lines = re.findall(rb"TN([0-9A-Fa-f]{2})([0-9A-Fa-f]{4})", bytes(sink))
    assert lines, f"no TN lines decoded; raw={bytes(sink)!r}"
    n = 0
    for cc, yy in lines:
        c = int(cc, 16)
        y = int(yy, 16)
        assert y == 2 * c, f"on-chip compute wrong: c={c} y={y} expected={2*c}"
        n += 1
    dut._log.info(f"PASS: arty_top decoded {n} UART lines, all y == 2*c (e.g. c={int(lines[0][0],16)})")
