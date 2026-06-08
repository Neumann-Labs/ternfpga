"""cocotb test: P-lane parallel ternary PE array, bit-exact vs NumPy.

Drives a shared activation vector + P weight rows and checks all P dot lanes
compute the right result simultaneously — the throughput path, still 0 DSP.
"""
import os
import random
import sys

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import pack_activations, pack_weights, ternary_dot_golden  # noqa: E402

K = int(os.environ.get("K", "8"))
P = int(os.environ.get("P", "4"))


def _to_signed32(raw: int) -> int:
    return raw - (1 << 32) if raw & (1 << 31) else raw


@cocotb.test()
async def pe_array(dut):
    rng = random.Random(0xA77A)
    trials = int(os.environ.get("TRIALS", "500"))
    for _ in range(trials):
        x = [rng.randint(-128, 127) for _ in range(K)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(P)]

        dut.a_flat.value = pack_activations(x)
        w_rows = 0
        for p in range(P):
            w_rows |= pack_weights(W[p]) << (2 * K * p)
        dut.w_rows.value = w_rows
        await Timer(1, units="ns")

        allbits = int(dut.dots.value)
        for p in range(P):
            got = _to_signed32((allbits >> (32 * p)) & 0xFFFFFFFF)
            exp = ternary_dot_golden(x, W[p])    # golden(activations, weights)
            assert got == exp, f"lane {p}: got={got} exp={exp}"

    dut._log.info(f"PASS: {trials} trials x P={P} parallel ternary dots bit-exact (K={K}, 0 DSP)")
