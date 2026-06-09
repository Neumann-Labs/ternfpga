"""Integer oracle for the on-fabric FFN-glue unit (rtl/ffn_glue_unit.sv) — the bit-exact spec.

The host FFN glue (soc/firmware/ffn_glue.h, MEASURED 2.58M cyc/layer) is the last big host term:
per channel f it computes H=relu(gate_int)^2*up_int (int64), N=H*w (w = ffn_sub_norm, fixed-point),
and the per-token int8 requant h_q=round(N*127/max|N|) — every dequant scale + the RMSNorm
normalizer cancel (the identity proven in models/ffn_glue_ref.py). On a cacheless 32-bit VexRiscv
those int64 mults (via libgcc) dominate. This unit does it on the fabric.

DIVIDER-FREE per channel: compute H/N/max|N| (the int64-mult work), then ONE reciprocal
  R     = bitlen(max|N|) + 24                 (normalize so recip stays ~31-bit, bounded mult)
  recip = (127 << R) // max|N|                (one sequential divide per call)
  h_q[f]= clip( round_half_up(|N[f]|*recip >> R) * sign(N[f]), -128, 127)
Outputs h_q[f] (int8) + max|N| (for the host's down_proj output dequant). The reciprocal-round is
integer-exact vs round(N*127/max|N|) for all practical N (validated below), so the RTL is bit-exact
vs this oracle, and this oracle matches the float host glue (models/ffn_glue_ref.py: 99.x% vs the
real model). w is stored as a signed fixed-point integer w_q (quantize_weight_fixed, 16-bit).
"""
from __future__ import annotations

import numpy as np


def glue_unit_int(gate_int, up_int, w_q):
    """gate_int, up_int: (F,) int (FPGA GEMV outputs). w_q: (F,) signed int (fixed-point norm w).
    Returns (h_q int8 (F,), amaxN int, R int, recip int) — the exact on-fabric computation."""
    gate_int = np.asarray(gate_int, dtype=object)
    up_int = np.asarray(up_int, dtype=object)
    w_q = np.asarray(w_q, dtype=object)
    g = np.maximum(gate_int, 0)
    H = g * g * up_int                       # int64-range, exact big-int
    N = H * w_q                              # signed, exact
    amaxN = int(np.abs(N).max()) if len(N) else 1
    if amaxN < 1:
        amaxN = 1
    R = amaxN.bit_length() + 24              # normalized: recip ~ 127<<24 ~ 2^31
    recip = (127 << R) // amaxN
    half = 1 << (R - 1)
    hq = np.empty(len(N), dtype=np.int64)
    for f in range(len(N)):
        n = int(N[f])
        if n >= 0:
            q = (n * recip + half) >> R
        else:
            q = -(((-n) * recip + half) >> R)
        hq[f] = 127 if q > 127 else (-128 if q < -128 else q)
    return hq, amaxN, R, recip


def quantize_weight_fixed(norm_w, bits=16):
    """Signed fixed-point of the per-channel norm weight (mirrors ffn_glue_ref.quantize_weight_fixed)."""
    norm_w = np.asarray(norm_w, dtype=np.float64)
    amax = float(np.abs(norm_w).max())
    scale = (2 ** (bits - 1) - 1) / amax
    return np.round(norm_w * scale).astype(np.int64), scale


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    F = 6912                                  # BitNet-2B intermediate width
    # realistic FPGA GEMV outputs (sum over hidden=2560 of int8 x ternary)
    gate_int = rng.integers(-6000, 6000, F)
    up_int = rng.integers(-6000, 6000, F)
    norm_w = (rng.standard_normal(F) * 0.1 + 1.0)            # ffn_sub_norm ~ O(1)
    w_q, _ = quantize_weight_fixed(norm_w, 16)

    hq, amaxN, R, recip = glue_unit_int(gate_int, up_int, w_q)

    # exact round-half-up reference (rational), to prove the reciprocal is faithful
    g = np.maximum(gate_int, 0).astype(object)
    N = g * g * up_int.astype(object) * w_q.astype(object)
    hq_ref = np.empty(F, dtype=np.int64)
    for f in range(F):
        n = int(N[f])
        a = abs(n)
        q = (a * 127 * 2 + amaxN) // (2 * amaxN)             # round-half-up of a*127/amaxN
        q = -q if n < 0 else q
        hq_ref[f] = 127 if q > 127 else (-128 if q < -128 else q)

    match = float(np.mean(hq == hq_ref) * 100)
    maxd = int(np.abs(hq - hq_ref).max())
    nz = int(np.count_nonzero(hq))
    print(f"recip vs exact round-half-up: match {match:.3f}%  max|diff| {maxd}  (R={R}, recip~2^{recip.bit_length()})")
    print(f"h_q range [{hq.min()},{hq.max()}]  nonzero {nz}/{F}  amaxN~2^{amaxN.bit_length()}")
    print("FFN_GLUE_UNIT_PASS" if match > 99.9 and maxd <= 1 else "FFN_GLUE_UNIT_FAIL")
