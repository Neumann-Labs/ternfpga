"""No-transcendental (LUT / fixed-point) attention glue — the de-risk for Phase 4.

The Phase-3 soft-float glue needed libm (cos/sin/exp/sqrt), which made the firmware binary
heavy and trapped on-board. The ONLY transcendentals in attention glue are RoPE's cos/sin and
softmax's exp; both can be **precomputed LUTs** (static tables, no libm). attn_sub_norm's 1/sqrt
cancels in the o_proj requant (the integer trick proven in ffn_glue). So the on-device glue needs
no libm at all — this module proves the LUTs hold accuracy vs the float golden (attn_ref) before
the C port.

  python models/glue_fixed_ref.py     # synthetic self-test vs attn_ref (cosine)
"""
from __future__ import annotations

import numpy as np

from ffn_ref import bitlinear, rms_norm

ROPE_Q = 15                 # cos/sin fixed-point bits (Q15 int16)
EXP_N, EXP_LSB = 4096, 0.01  # exp LUT: 4096 entries over logit-drop [0, 40.96]


def build_rope_lut(max_pos, head_dim, theta):
    """Precomputed cos/sin as Q15 ints — a static ROM table, built with np (host), no on-device libm."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.arange(max_pos), inv_freq)            # (max_pos, head_dim/2)
    scale = float(1 << ROPE_Q)
    return np.round(np.cos(ang) * scale).astype(np.int64), np.round(np.sin(ang) * scale).astype(np.int64)


def apply_rope_lut(x, positions, cos_q, sin_q):
    """x: (n_heads, T, head_dim). Fixed-point rotation using the Q15 LUT (>>Q15 == /2^15)."""
    half = x.shape[-1] // 2
    c = cos_q[positions].astype(np.float64) / (1 << ROPE_Q)   # (T, half) — simulates Q15 fixed-point
    s = sin_q[positions].astype(np.float64) / (1 << ROPE_Q)
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * c[None] - x2 * s[None]
    out[..., half:] = x2 * c[None] + x1 * s[None]
    return out


def build_exp_lut():
    """exp(-idx*LSB) table — a static ROM (built with np, host), indexed by the quantized logit drop."""
    return np.exp(-np.arange(EXP_N, dtype=np.float64) * EXP_LSB)


def softmax_fixed(scores, exp_lut):
    """Causal softmax via the exp LUT: quantize (max - logit) to an index, look up exp, normalize."""
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    s = np.where(mask[None], -np.inf, scores)
    s = s - s.max(axis=-1, keepdims=True)                    # <= 0
    idx = np.clip(np.round(-s / EXP_LSB), 0, EXP_N - 1).astype(np.int64)
    e = exp_lut[idx]
    e = np.where(mask[None], 0.0, e)
    return e / e.sum(axis=-1, keepdims=True)


def attention_fixed(x, Wq, sq, Wk, sk, Wv, sv, Wo, so, attn_norm_weight,
                    n_heads, n_kv_heads, head_dim, theta, eps, positions=None):
    """Attention with LUT RoPE + LUT softmax (no transcendentals). Same signature as attn_ref.attention."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    T = x.shape[0]
    if positions is None:
        positions = np.arange(T)
    n_rep = n_heads // n_kv_heads

    q, _, _ = bitlinear(x, Wq, sq)
    k, _, _ = bitlinear(x, Wk, sk)
    v, _, _ = bitlinear(x, Wv, sv)
    qh = q.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
    kh = k.reshape(T, n_kv_heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(T, n_kv_heads, head_dim).transpose(1, 0, 2)

    cos_q, sin_q = build_rope_lut(int(np.max(positions)) + 1, head_dim, theta)
    qh = apply_rope_lut(qh, positions, cos_q, sin_q)
    kh = apply_rope_lut(kh, positions, cos_q, sin_q)
    kh = np.repeat(kh, n_rep, axis=0)
    vh = np.repeat(vh, n_rep, axis=0)

    scores = np.einsum("htd,hsd->hts", qh, kh) * (1.0 / np.sqrt(head_dim))
    a = softmax_fixed(scores, build_exp_lut())
    ctx = np.einsum("hts,hsd->htd", a, vh)
    o = ctx.transpose(1, 0, 2).reshape(T, n_heads * head_dim)
    o = rms_norm(o, attn_norm_weight, eps)                   # cancels into o_proj requant on-device
    y, _, _ = bitlinear(o, Wo, so)
    return y, {}


if __name__ == "__main__":
    from attn_ref import attention as attention_float

    T, HID, NH, NKV, HD = 16, 256, 8, 2, 32
    rng = np.random.default_rng(7)

    def tern(o, i):
        return rng.choice([-1, 0, 1], size=(o, i)).astype(np.int64)

    x = rng.standard_normal((T, HID)) * 0.6
    Wq, Wk, Wv, Wo = tern(HID, HID), tern(NKV * HD, HID), tern(NKV * HD, HID), tern(HID, HID)
    sq, sk, sv, so = 1.21875, 1.79688, 2.29688, 0.96484
    nw = rng.uniform(0.5, 1.5, size=HID)

    yf, _ = attention_float(x, Wq, sq, Wk, sk, Wv, sv, Wo, so, nw, NH, NKV, HD, 500000.0, 1e-5)
    yx, _ = attention_fixed(x, Wq, sq, Wk, sk, Wv, sv, Wo, so, nw, NH, NKV, HD, 500000.0, 1e-5)
    cos = float(np.dot(yf.flatten(), yx.flatten()) / (np.linalg.norm(yf) * np.linalg.norm(yx) + 1e-12))
    max_abs = float(np.abs(yx - yf).max())
    # rel err only over non-negligible elements (near-zero refs blow up a naive rel err)
    big = np.abs(yf) > 0.1 * np.abs(yf).max()
    rel_big = float(np.mean(np.abs(yx - yf)[big] / np.abs(yf)[big]))
    print(f"LUT RoPE(Q{ROPE_Q}) + LUT softmax(N={EXP_N},lsb={EXP_LSB}) vs float golden:")
    print(f"  cosine={cos:.6f}  max_abs_err={max_abs:.3e}  rel_err(|y|>10%max)={rel_big:.3e}")
    print("FIXED_GLUE_GOLDEN_PASS" if cos > 0.9999 and rel_big < 0.02 else "FIXED_GLUE_GOLDEN_FAIL")
