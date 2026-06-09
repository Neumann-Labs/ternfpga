"""Fixed-point FFN glue — the on-chip path that needs NO float dequant / NO RMSNorm divide.

Key identity (derived below, verified in __main__): the int8 down_proj input `h_q`
depends ONLY on the INTEGER quantity N_i, because every per-token dequant scale and
the RMSNorm normalizer CANCEL in the final requant:

    H_i  = relu(gate_int_i)^2 * up_int_i        # integer; gate_int/up_int = the FPGA GEMV outputs
    N_i  = H_i * w_i                            # w_i = ffn_sub_norm weight (the only non-integer)
    h_q_i = round( N_i * 127 / max_j|N_j| )      # per-token requant -> int8

Proof. ffn_ref's glue gives h_pre = C*H with C = (sg*amax_x/127)^2 * (su*amax_x/127) a
per-token constant. RMSNorm: h = h_pre / sqrt(mean(h_pre^2)+eps) * w = C*H*w / (C*rms(H)) =
H*w/rms(H) = N/rms(H)  (C cancels; eps dropped as negligible vs mean(H^2)). The down_proj
int8 quant scale is 127/max|h| = 127*rms(H)/max|N|, so
    h_q = round(h * scale) = round( (N/rms(H)) * 127*rms(H)/max|N| ) = round(N * 127/max|N|).
Both C and rms(H) cancel. => the FPGA produces h_q with integer multiplies + ONE per-token
reciprocal (127/max|N|); the host only applies the final per-token *output* scale to down_int.

This is the spec for an on-chip glue unit (relu^2 + elementwise via DSPs + absmax requant),
keeping the gate/up -> down path entirely on-chip (no soft-CPU round-trip — the latency risk
flagged in docs/research/scaling-feasibility.md). `w_i` is quantized to fixed-point for the
hardware; __main__ checks both the float-w identity and a fixed-point-w variant vs ffn_ref.
"""
from __future__ import annotations

import numpy as np


def glue_hq(gate_int, up_int, norm_w):
    """gate_int, up_int: (rows, ff) int (the FPGA GEMV outputs). norm_w: (ff,) float|fixed.
    Returns (h_q int8 (rows,ff), amaxN float (rows,1)). h_q is the down_proj int8 input."""
    gate_int = np.asarray(gate_int, dtype=np.int64)
    up_int = np.asarray(up_int, dtype=np.int64)
    H = np.maximum(gate_int, 0).astype(object) ** 2 * up_int.astype(object)   # exact big-int
    N = H.astype(np.float64) * np.asarray(norm_w, dtype=np.float64)
    amaxN = np.maximum(np.abs(N).max(axis=-1, keepdims=True), 1e-9)
    h_q = np.clip(np.round(N * 127.0 / amaxN), -128, 127).astype(np.int64)
    return h_q, amaxN


def quantize_weight_fixed(norm_w, bits=16):
    """Quantize the per-channel norm weight to a signed fixed-point (bits) for the hardware."""
    norm_w = np.asarray(norm_w, dtype=np.float64)
    amax = np.abs(norm_w).max()
    scale = (2 ** (bits - 1) - 1) / amax
    q = np.round(norm_w * scale)
    return q, scale                       # effective w_i ~= q/scale; the 1/scale folds into amaxN


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ffn_ref import ffn_block

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    MODEL = "microsoft/BitNet-b1.58-2B-4T"
    tok = AutoTokenizer.from_pretrained(MODEL)
    load_kw = {} if args.device == "cpu" else {"device_map": args.device}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, **load_kw).eval()
    mlp = model.model.layers[args.layer].mlp

    cap = {}

    def hook(_m, inp, out):
        cap["x"] = inp[0].detach().float().cpu().numpy()[0]

    h = mlp.register_forward_hook(hook)
    ids = tok("Ternary weights make the multiply a select, and sparsity skips the fetch.",
              return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)
    h.remove()

    def ext(lin):
        return lin.weight.detach().float().cpu().numpy(), \
               float(lin.weight_scale.detach().float().cpu().numpy().reshape(-1)[0])

    Wg, sg = ext(mlp.gate_proj)
    Wu, su = ext(mlp.up_proj)
    Wd, sd = ext(mlp.down_proj)
    fn = mlp.ffn_sub_norm.weight.detach().float().cpu().numpy()
    eps = getattr(model.config, "rms_norm_eps", 1e-5)

    # ffn_ref is the validated reference; it exposes the integer GEMV outputs + its own h_q
    _, inter = ffn_block(cap["x"], Wg, sg, Wu, su, Wd, sd, fn, eps)
    ref_hq = inter["down_xq"]                      # ffn_ref's int8 down_proj input

    # our integer-only glue, float w
    my_hq, _ = glue_hq(inter["gate_int"], inter["up_int"], fn)
    match = np.mean(my_hq == ref_hq) * 100
    maxd = int(np.abs(my_hq - ref_hq).max())
    print(f"float-w  : h_q exact-match {match:.2f}%   max|diff| {maxd}")

    # fixed-point w (what the hardware stores)
    for bits in (12, 16):
        qw, _ = quantize_weight_fixed(fn, bits)
        fx_hq, _ = glue_hq(inter["gate_int"], inter["up_int"], qw)
        m = np.mean(fx_hq == ref_hq) * 100
        d = int(np.abs(fx_hq - ref_hq).max())
        print(f"fixed w{bits}: h_q exact-match {m:.2f}%   max|diff| {d}")
    print("OK — the dequant/RMSNorm-cancellation identity holds; on-chip integer glue is justified"
          if match > 99 else "MISMATCH — revisit the derivation")
