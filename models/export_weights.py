"""Export ternary model weights to the packed format the RTL consumes.

The bridge from a trained BitNet b1.58 layer to ternfpga: apply the BitNet
absmean ternarization, pack weights to the 2-bit little-endian codes that
`ternary_ref.py` (and the RTL) define, and read tiles back. Encoding lives in
exactly one place (`ternary_ref.py`); this module reuses it so model export and
hardware can never disagree.

Self-test:  python models/export_weights.py
"""
from __future__ import annotations
import numpy as np

from ternary_ref import pack_weights, pack_activations  # one source of truth for the encoding


def ternarize_absmean(W, eps: float = 1e-5):
    """BitNet b1.58 weight quantization: scale = mean(|W|); Wq = clamp(round(W/scale), -1, 1).

    Returns (Wq int8 in {-1,0,+1}, scale float). This is the exact rule the model
    is trained with, so applying it to the stored latent weights reproduces the
    ternary weights the model actually uses at inference.
    """
    W = np.asarray(W, dtype=np.float64)
    scale = float(np.mean(np.abs(W))) + eps
    Wq = np.clip(np.round(W / scale), -1, 1).astype(np.int8)
    return Wq, scale


def pack_rows(W_tern):
    """Pack an MxK ternary matrix into a list of per-row ints (the RTL `w_row` bus)."""
    return [pack_weights([int(v) for v in row]) for row in np.asarray(W_tern)]


def sparsity(W_tern) -> float:
    """Fraction of zero weights (the activation-/weight-sparsity lever, on real data)."""
    return float(np.mean(np.asarray(W_tern) == 0))


def save_tile(path, W_tern, x):
    """Persist a ternary weight tile + int8 activation vector as .npz for the cocotb tests."""
    np.savez(path,
             W=np.asarray(W_tern, dtype=np.int8),
             x=np.asarray(x, dtype=np.int64))


def load_tile(path):
    """Load a tile saved by `save_tile`. Returns (W int64 MxK ternary, x int64 K)."""
    d = np.load(path)
    return d["W"].astype(np.int64), d["x"].astype(np.int64)


def _unpack_row(packed: int, k: int):
    out = []
    for i in range(k):
        code = (packed >> (2 * i)) & 0b11
        out.append(1 if code == 1 else (-1 if code == 2 else 0))
    return out


# ---- Dense base-3 packing: 5 ternary weights per byte (1.6 bits/weight) ----
# 3**5 = 243 < 256, so five trits fit in a byte: byte = sum(t_i * 3**i), with
# trit 0->0, 1->+1, 2->-1. 20% tighter than the 2-bit codes, 5x tighter than
# INT8 — matches rtl/ternary_unpack5.sv exactly.

def _w_to_trit(w: int) -> int:
    return 0 if w == 0 else (1 if w == 1 else 2)


def pack_trits5(w5) -> int:
    """Pack up to 5 ternary weights into one base-3 byte (little-endian trits)."""
    assert len(w5) <= 5, "at most 5 ternary weights per byte"
    return sum(_w_to_trit(int(w)) * (3 ** i) for i, w in enumerate(w5))


def unpack_trits5(byte: int, n: int = 5):
    """Inverse of pack_trits5: byte -> list of n ternary weights {-1,0,+1}."""
    b, out = int(byte), []
    for _ in range(n):
        t = b % 3
        b //= 3
        out.append(0 if t == 0 else (1 if t == 1 else -1))
    return out


def trit_codes5(byte: int) -> int:
    """byte -> 10-bit value of 5 x 2-bit ternary codes (lane 0 in the LSBs),
    matching rtl/ternary_unpack5.sv's codes_out output."""
    codes = 0
    for i, w in enumerate(unpack_trits5(byte)):
        c = 0b01 if w == 1 else (0b10 if w == -1 else 0b00)
        codes |= c << (2 * i)
    return codes


def pack_row_trits5(row) -> bytes:
    """Pack a ternary row into ceil(len/5) base-3 bytes (the dense DDR3 layout)."""
    row = [int(v) for v in row]
    return bytes(pack_trits5(row[i:i + 5]) for i in range(0, len(row), 5))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.normal(size=(16, 8))
    Wq, scale = ternarize_absmean(W)
    assert set(np.unique(Wq)).issubset({-1, 0, 1}), "ternarize must yield {-1,0,1}"
    # pack/unpack round-trip must be exact
    for row in Wq:
        packed = pack_weights([int(v) for v in row])
        assert _unpack_row(packed, len(row)) == [int(v) for v in row], "pack/unpack mismatch"
    # activation pack round-trip
    a = [int(v) for v in rng.integers(-128, 128, size=8)]
    av = pack_activations(a)
    assert [((av >> (8 * i)) & 0xFF) - 256 if ((av >> (8 * i)) & 0xFF) & 0x80 else ((av >> (8 * i)) & 0xFF)
            for i in range(8)] == a
    # base-3 dense packing (5 trits/byte) — exhaustive round-trip over all 243 bytes
    for byte in range(243):
        w5 = unpack_trits5(byte)
        assert pack_trits5(w5) == byte, f"trit round-trip failed at {byte}"
        assert trit_codes5(byte) == pack_weights(w5), f"trit_codes5 != pack_weights at {byte}"
    print(f"export_weights self-test OK: ternary {{-1,0,1}}, pack/unpack exact, "
          f"base-3 5-trit/byte round-trip exact (1.6 bits/weight), "
          f"absmean scale={scale:.4f}, sparsity={sparsity(Wq):.2f}")
