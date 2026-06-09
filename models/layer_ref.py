"""Golden reference for one full BitNet b1.58 decoder layer — composes the attention
(attn_ref) + FFN (ffn_ref) goldens with the two RMSNorms and the residuals, exactly as
microsoft/BitNet-b1.58-2B-4T (LLaMA-style pre-norm):

    h = x + attention( input_layernorm(x) )
    y = h + ffn_block( post_attention_layernorm(h) )

The FPGA runs the 7 ternary GEMVs (q/k/v/o + gate/up/down); the host (VexRiscv) runs the
norms, residuals, RoPE/softmax (attn glue) and the relu^2/mul/requant (FFN glue). This is
the per-layer oracle for the on-board single-layer test (#30) and the decode loop (#31).
"""
from __future__ import annotations

import numpy as np

from attn_ref import attention
from ffn_ref import ffn_block, rms_norm


def decoder_layer(x, w, cfg):
    """One decoder layer. x: (T, hidden). w: dict of weights/scales. cfg: dict of dims.
    Returns (y float (T,hidden), intermediates dict)."""
    eps = cfg["eps"]
    h_in = rms_norm(x, w["input_ln"], eps)
    attn_out, attn_i = attention(
        h_in, w["Wq"], w["sq"], w["Wk"], w["sk"], w["Wv"], w["sv"], w["Wo"], w["so"],
        w["attn_norm"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
        cfg["theta"], eps)
    h = x + attn_out                                       # residual 1

    h_post = rms_norm(h, w["post_ln"], eps)
    ffn_out, ffn_i = ffn_block(
        h_post, w["Wg"], w["sg"], w["Wu"], w["su"], w["Wd"], w["sd"], w["ffn_norm"], eps)
    y = h + ffn_out                                        # residual 2

    return y, {"attn_out": attn_out, "ffn_out": ffn_out, "h_mid": h,
               "attn": attn_i, "ffn": ffn_i}
