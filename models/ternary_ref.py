"""Golden reference for the ternary datapath (bit-exact oracle for the RTL).

Ternary weights are encoded for the hardware as 2-bit codes:
    +1 -> 0b01 (1)
    -1 -> 0b10 (2)
     0 -> 0b00 (0)   (code 0b11 is unused/reserved)
Activations are signed 8-bit (two's complement), packed little-endian.
"""
from __future__ import annotations
import numpy as np


def ternary_dot_golden(a, w) -> int:
    """Exact integer dot product of int8 activations `a` and ternary weights `w`."""
    a = np.asarray(a, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    assert set(np.unique(w)).issubset({-1, 0, 1}), "weights must be ternary"
    return int(np.dot(a, w))


def pack_activations(a) -> int:
    """Pack K signed int8 activations little-endian into one integer bus value."""
    v = 0
    for i, x in enumerate(a):
        v |= (int(x) & 0xFF) << (8 * i)
    return v


def weight_code(x: int) -> int:
    return 1 if x == 1 else (2 if x == -1 else 0)


def pack_weights(w) -> int:
    """Pack K ternary weights little-endian as 2-bit codes into one integer bus value."""
    v = 0
    for i, x in enumerate(w):
        v |= weight_code(x) << (2 * i)
    return v


def ternary_gemv_golden(W, x):
    """Exact y[m] = sum_k W[m,k]*x[k] for ternary W (M x K) and int8 x (K). Returns list[int]."""
    W = np.asarray(W, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    assert W.size == 0 or set(np.unique(W)).issubset({-1, 0, 1}), "weights must be ternary"
    return (W @ x).tolist()


def ternary_gemv_sparse_golden(W, x, mask):
    """Activation-sparse GEMV: y[m] = dot(W[m], x) if mask[m] else 0 (inactive rows skipped)."""
    full = ternary_gemv_golden(W, x)
    return [full[m] if mask[m] else 0 for m in range(len(full))]

