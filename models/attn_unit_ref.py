"""Bit-exact INTEGER attention-unit oracle — the spec the RTL (rtl/attention_unit.sv) matches.

For one query vs a KV cache of T keys (D=head_dim), in exactly the integer arithmetic the
hardware does:
    scores[j] = sum_d q[d]*k[j][d]                         # int dot (DSP/LUT MACs)
    drop[j]   = ((max_s - scores[j]) * score_scale_q) >> SHIFT     # logit drop -> LUT index
    e[j]      = EXP_LUT[clip(drop[j], 0, EXP_N-1)]         # Q15 exp, from BRAM
    ctx[d]    = ( sum_j e[j]*v[j][d] ) // sum_j e[j]       # integer normalize

score_scale_q/SHIFT encode the softmax temperature (q_scale*k_scale/sqrt(d)) as a fixed-point
CSR. The fixed-point LUT approach is already validated end-to-end (glue_fixed_ref, cosine
0.999999); this module is the bit-exact oracle for the cocotb testbench + a cosine sanity check.

  python models/attn_unit_ref.py     # self-test vs float softmax-attention
"""
from __future__ import annotations

import numpy as np

EXP_N, EXP_LSB, QBITS, SHIFT = 4096, 0.01, 15, 24


def exp_lut():
    # clip to 32767 so e fits signed-16 in hardware (exp(0)*2^15 = 32768 would read as -32768);
    # matches soc/firmware/gen_glue_luts.py. Negligible (1 part in 32768).
    return np.clip(np.round(np.exp(-np.arange(EXP_N) * EXP_LSB) * (1 << QBITS)), 0, 32767).astype(np.int64)


def attn_unit_int(q_i, K_i, V_i, score_shift, EL):
    """Exact integer attention for one head. q_i:(D,), K_i/V_i:(T,D) int. Returns ctx:(D,) int.
    The softmax temperature is a single programmable RIGHT SHIFT (no multiply): the exp-LUT index
    is idx = (max_score - score) >> score_shift, with score_shift matched to EXP_LSB by the host."""
    q_i = q_i.astype(np.int64); K_i = K_i.astype(np.int64); V_i = V_i.astype(np.int64)
    scores = K_i @ q_i                                       # (T,) int64
    m = int(scores.max())
    idx = np.clip((m - scores) >> score_shift, 0, EXP_N - 1).astype(np.int64)
    e = EL[idx]                                              # (T,) Q15
    s = int(e.sum())
    num = (e[:, None] * V_i).sum(axis=0)                     # (D,) int64 — raw weighted sum (RTL output)
    ctx = (num // max(s, 1)).astype(np.int64)                # final /sum (trivial host step)
    return ctx, scores, e, s, num.astype(np.int64)


def _attn_float(q, K, V, scale):
    sc = (K @ q) * scale
    sc = sc - sc.max()
    p = np.exp(sc); p = p / p.sum()
    return p @ V


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    T, D = 64, 128
    q = rng.standard_normal(D) * 0.5
    K = rng.standard_normal((T, D)) * 0.5
    V = rng.standard_normal((T, D)) * 0.5

    # int16 quant: the unit consumes the projection+RoPE INTEGER outputs (int16-range, not
    # freshly int8-quantized) — so it has ~15-bit precision for the scores/a@V MACs.
    qs = np.abs(q).max() / 32767.0
    ks = np.abs(K).max() / 32767.0
    vs = np.abs(V).max() / 32767.0
    q_i = np.round(q / qs).astype(np.int64)
    K_i = np.round(K / ks).astype(np.int64)
    V_i = np.round(V / vs).astype(np.int64)

    # idx = (max-score) >> score_shift; matched so one idx step ~= EXP_LSB of real logit:
    #   real logit = scores_int * (qs*ks/sqrt(D)); idx = drop_int / 2^shift = drop_real/EXP_LSB
    #   -> 2^shift = EXP_LSB / (qs*ks/sqrt(D))
    score_shift = int(round(np.log2(EXP_LSB * np.sqrt(D) / (qs * ks))))

    EL = exp_lut()
    ctx_i, scores, e, s, num = attn_unit_int(q_i, K_i, V_i, score_shift, EL)
    ctx_int_real = ctx_i.astype(np.float64) * vs                 # dequant the output
    ctx_float = _attn_float(q, K, V, 1.0 / np.sqrt(D))

    cos = float(np.dot(ctx_int_real, ctx_float) /
                (np.linalg.norm(ctx_int_real) * np.linalg.norm(ctx_float) + 1e-12))
    big = np.abs(ctx_float) > 0.1 * np.abs(ctx_float).max()
    rel = float(np.mean(np.abs(ctx_int_real - ctx_float)[big] / np.abs(ctx_float)[big]))
    print(f"integer attention unit (T={T} D={D}, exp LUT N={EXP_N}) vs float softmax-attention:")
    print(f"  score_shift={score_shift}  cosine={cos:.6f}  rel_err(|y|>10%max)={rel:.3e}")
    print("ATTN_UNIT_ORACLE_PASS" if cos > 0.999 and rel < 0.05 else "ATTN_UNIT_ORACLE_FAIL")
