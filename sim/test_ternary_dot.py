"""cocotb test: the ternary dot-product RTL must be bit-exact vs the NumPy golden.

This is the project's foundational claim in miniature: the multiply-free
sign-select + adder datapath computes integer dot products exactly. Run via
`make -C sim` (verilator backend).
"""
import os
import random
import sys

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from ternary_ref import ternary_dot_golden, pack_activations, pack_weights  # noqa: E402

K = int(os.environ.get("K", "8"))
N_ITERS = int(os.environ.get("N_ITERS", "2000"))


def _to_signed(value, width=32) -> int:
    """Read a cocotb handle value as a signed int, across cocotb versions."""
    try:
        return int(value.to_signed())
    except AttributeError:
        v = int(value)
        return v - (1 << width) if v & (1 << (width - 1)) else v


@cocotb.test()
async def random_dot_products(dut):
    rng = random.Random(0xC0FFEE)

    # Directed edge cases first.
    edge = [
        ([0] * K, [0] * K),
        ([127] * K, [1] * K),
        ([-128] * K, [1] * K),
        ([127] * K, [-1] * K),
        ([-128] * K, [-1] * K),
        ([rng.randint(-128, 127) for _ in range(K)], [0] * K),
    ]
    for a, w in edge:
        dut.a_flat.value = pack_activations(a)
        dut.w_flat.value = pack_weights(w)
        await Timer(2, "ns")
        got, exp = _to_signed(dut.dot.value), ternary_dot_golden(a, w)
        assert got == exp, f"edge: a={a} w={w} got={got} exp={exp}"

    # Randomized fuzz.
    for n in range(N_ITERS):
        a = [rng.randint(-128, 127) for _ in range(K)]
        w = [rng.choice([-1, 0, 1]) for _ in range(K)]
        dut.a_flat.value = pack_activations(a)
        dut.w_flat.value = pack_weights(w)
        await Timer(2, "ns")
        got, exp = _to_signed(dut.dot.value), ternary_dot_golden(a, w)
        assert got == exp, f"iter {n}: a={a} w={w} got={got} exp={exp}"

    dut._log.info(f"PASS: {len(edge)} edge + {N_ITERS} random K={K} ternary dot-products bit-exact vs numpy")
