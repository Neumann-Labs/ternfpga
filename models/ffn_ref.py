"""Golden reference for the BitNet b1.58 FFN block — bit-exact integer matmuls + host glue.

Derived from the real microsoft/BitNet-b1.58-2B-4T (see inspect_bitnet_ffn.py):

    MLP(x) = down_proj( ffn_sub_norm( ReLU(gate_proj(x))^2 * up_proj(x) ) )

Each *_proj is an AutoBitLinear. For input row x (one token):
    scale_x = 127 / max(|x|)                       # per-token absmax (int8)
    x_q     = round(x * scale_x).clip(-128, 127)   # int8  <-- the FPGA matmul input
    y_int   = x_q @ W_tern^T                        # int32 <-- THE FPGA OP (0-DSP ternary)
    y       = y_int * weight_scale / scale_x        # dequant (host), == y_int * weight_scale * max|x| / 127

The FPGA computes the three `int8 x ternary -> int32` matmuls; the host does the per-token
quant, the dequant scale, ReLU^2, the elementwise multiply, and the RMSNorm (ffn_sub_norm).
This is the *system* oracle; `ternary_ref.ternary_gemv_golden` is the bit-exact oracle for the
integer matmul the RTL implements. Intermediates (int8 inputs, int32 outputs) are returned so
the RTL testbench can drive exactly the FPGA boundary.
"""
from __future__ import annotations

import numpy as np

from ternary_ref import ternary_gemv_golden


def act_quant_int8(x, eps: float = 1e-5):
    """Per-token symmetric int8 quant (BitNet ActQuant). x: (rows, in).
    Returns (x_q int64 in [-128,127], amax float (rows,1))."""
    x = np.asarray(x, dtype=np.float64)
    amax = np.maximum(np.abs(x).max(axis=-1, keepdims=True), eps)
    x_q = np.clip(np.round(x * (127.0 / amax)), -128, 127).astype(np.int64)
    return x_q, amax


def bitlinear(x, W_tern, weight_scale):
    """One AutoBitLinear. x: (rows, in); W_tern: (out, in) in {-1,0,1}; weight_scale: scalar.
    Returns (y float (rows,out), y_int int32 (rows,out), x_q int8 (rows,in))."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    W_tern = np.asarray(W_tern, dtype=np.int64)
    x_q, amax = act_quant_int8(x)
    y_int = np.stack([np.asarray(ternary_gemv_golden(W_tern, x_q[r]), dtype=np.int64)
                      for r in range(x_q.shape[0])])           # the FPGA op, per token
    y = y_int.astype(np.float64) * (weight_scale * amax / 127.0)
    return y, y_int, x_q


def rms_norm(x, weight, eps):
    """RMSNorm (BitNetRMSNorm): x / sqrt(mean(x^2)+eps) * weight. Preserves zeros."""
    x = np.asarray(x, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * np.asarray(weight, dtype=np.float64)


def relu2(x):
    """ReLUSquaredActivation: relu(x)^2."""
    r = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return r * r


def ffn_block(x, Wg, sg, Wu, su, Wd, sd, ffn_norm_weight, eps):
    """BitNet FFN block. Returns (y float (rows,hidden), intermediates dict).

    intermediates exposes the FPGA boundary for the RTL testbench:
      gate_xq/up_xq/down_xq  — int8 matmul inputs
      gate_int/up_int/down_int — int32 matmul outputs
    """
    gate, gate_int, gate_xq = bitlinear(x, Wg, sg)
    up, up_int, up_xq = bitlinear(x, Wu, su)
    h = relu2(gate) * up
    h = rms_norm(h, ffn_norm_weight, eps)
    y, down_int, down_xq = bitlinear(h, Wd, sd)
    return y, {
        "gate_xq": gate_xq, "up_xq": up_xq, "down_xq": down_xq,
        "gate_int": gate_int, "up_int": up_int, "down_int": down_int,
        "h_pre_norm": relu2(gate) * up, "h": h,
    }
