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
    print(f"export_weights self-test OK: ternary {{-1,0,1}}, pack/unpack exact, "
          f"absmean scale={scale:.4f}, sparsity={sparsity(Wq):.2f}")
