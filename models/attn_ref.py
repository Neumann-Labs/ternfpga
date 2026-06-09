"""Golden reference for the BitNet b1.58 ATTENTION block — bit-exact ternary projections
+ host glue, mirroring microsoft/BitNet-b1.58-2B-4T (see inspect_bitnet_attn.py).

BitNetAttention.forward (confirmed from the real model):
    q = q_proj(x).view(T, n_heads,    head_dim)     # AutoBitLinear  (2560 -> 2560)
    k = k_proj(x).view(T, n_kv_heads, head_dim)     # AutoBitLinear  (2560 -> 640)  GQA
    v = v_proj(x).view(T, n_kv_heads, head_dim)     # AutoBitLinear  (2560 -> 640)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)     # RoPE, theta=500000
    k, v = repeat_kv(k, v, n_rep=4)                 # GQA: 5 kv heads -> 20
    scores = q @ k^T * (1/sqrt(head_dim)) + causal_mask ; a = softmax(scores)
    o = (a @ v).reshape(T, hidden)
    o = attn_sub_norm(o)                            # BitNetRMSNorm  <-- diff with Llama
    y = o_proj(o)                                   # AutoBitLinear  (2560 -> 2560)

The FPGA computes the four `int8 x ternary -> int32` projection matmuls (q/k/v/o); the host
(VexRiscv) does the per-token quant/dequant, RoPE, scores, softmax, the a@v weighted sum, and
the attn_sub_norm. Intermediates expose the FPGA boundary for the RTL testbench.
"""
from __future__ import annotations

import numpy as np

from ffn_ref import bitlinear, rms_norm


def rope_tables(positions, head_dim, theta):
    """HF LlamaRotaryEmbedding cos/sin. positions: (T,). Returns cos,sin: (T, head_dim)."""
    positions = np.asarray(positions, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = np.outer(positions, inv_freq)                 # (T, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)         # (T, head_dim)
    return np.cos(emb), np.sin(emb)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope(x, cos, sin):
    """x: (n_heads, T, head_dim); cos/sin: (T, head_dim) broadcast over heads."""
    return x * cos[None, :, :] + _rotate_half(x) * sin[None, :, :]


def _softmax_causal(scores):
    """scores: (n_heads, T, T). Causal mask + row softmax in float64."""
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)      # True above diagonal -> -inf
    s = np.where(mask[None], -np.inf, scores)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def attention(x, Wq, sq, Wk, sk, Wv, sv, Wo, so, attn_norm_weight,
              n_heads, n_kv_heads, head_dim, theta, eps, positions=None):
    """Full-sequence (prefill) BitNet attention. x: (T, hidden).
    Returns (y float (T,hidden), intermediates dict exposing the FPGA boundary)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    T = x.shape[0]
    if positions is None:
        positions = np.arange(T)
    n_rep = n_heads // n_kv_heads

    q, q_int, q_xq = bitlinear(x, Wq, sq)                 # (T, n_heads*head_dim)
    k, k_int, k_xq = bitlinear(x, Wk, sk)                 # (T, n_kv*head_dim)
    v, v_int, v_xq = bitlinear(x, Wv, sv)

    qh = q.reshape(T, n_heads, head_dim).transpose(1, 0, 2)     # (n_heads, T, d)
    kh = k.reshape(T, n_kv_heads, head_dim).transpose(1, 0, 2)  # (n_kv, T, d)
    vh = v.reshape(T, n_kv_heads, head_dim).transpose(1, 0, 2)

    cos, sin = rope_tables(positions, head_dim, theta)
    qh = apply_rope(qh, cos, sin)
    kh = apply_rope(kh, cos, sin)

    kh = np.repeat(kh, n_rep, axis=0)                     # GQA: (n_heads, T, d)
    vh = np.repeat(vh, n_rep, axis=0)

    scaling = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("htd,hsd->hts", qh, kh) * scaling  # (n_heads, T, T)
    a = _softmax_causal(scores)
    ctx = np.einsum("hts,hsd->htd", a, vh)                # (n_heads, T, d)
    o = ctx.transpose(1, 0, 2).reshape(T, n_heads * head_dim)   # (T, hidden)

    o = rms_norm(o, attn_norm_weight, eps)               # attn_sub_norm
    y, o_int, o_xq = bitlinear(o, Wo, so)

    return y, {
        "q_xq": q_xq, "k_xq": k_xq, "v_xq": v_xq, "o_xq": o_xq,
        "q_int": q_int, "k_int": k_int, "v_int": v_int, "o_int": o_int,
        "attn_pre_oproj": o,
    }
