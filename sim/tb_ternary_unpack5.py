"""cocotb test: dense base-3 weight unpacker (5 ternary weights / byte).

Exhaustively checks all 243 valid packed bytes — the RTL combinational decode
must equal the Python golden (`export_weights.trit_codes5`), and the recovered
ternary must round-trip back to the byte. Proves the dense 1.6-bit/weight layout
feeds the multiply-free lanes correctly.
"""
import os
import sys

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from export_weights import pack_trits5, trit_codes5, unpack_trits5  # noqa: E402


@cocotb.test()
async def unpack_all_bytes(dut):
    for byte in range(243):
        dut.byte_in.value = byte
        await Timer(1, units="ns")
        got = int(dut.codes_out.value)
        exp = trit_codes5(byte)
        assert got == exp, f"byte {byte}: got={got:#012b} exp={exp:#012b}"
        assert pack_trits5(unpack_trits5(byte)) == byte, f"round-trip failed at {byte}"
    dut._log.info("PASS: all 243 base-3 bytes unpack bit-exact to ternary codes "
                  "(5 weights/byte, 1.6 bits/weight)")
