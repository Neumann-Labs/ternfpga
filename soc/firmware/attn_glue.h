// Host (VexRiscv) glue for BitNet attention — mirrors models/attn_ref.py.
//
// The FPGA computes the four int8xternary->int32 projection matmuls (q/k/v/o). This
// header is the glue between q/k/v and o_proj for ONE autoregressive decode step:
//   dequant q/k/v -> RoPE(q,k) -> append k,v to the KV cache -> per-head causal
//   scores QK^T/sqrt(d) -> softmax -> a@V -> concat -> attn_sub_norm -> requant to o_q.
// Unlike the FFN glue this is NOT integer-only (softmax/RoPE are float), so it matches
// the golden to float precision, not bit-for-bit. Soft-float on VexRiscv is slow but
// one-time per token — an on-chip glue unit is the documented optimization.
#ifndef ATTN_GLUE_H
#define ATTN_GLUE_H
#include <stdint.h>
#include <math.h>

typedef struct {
	int hidden, n_heads, n_kv_heads, head_dim, max_seq;
	double theta, eps;
	double sq, sk, sv, so;        // projection weight_scales
	const double *norm_w;         // attn_sub_norm weight [hidden]
} attn_cfg_t;

// RoPE in place on one head-dim vector at `pos` (HF rotate_half convention).
static inline void attn_rope(float *x, int pos, int head_dim, double theta)
{
	int half = head_dim / 2;
	for (int i = 0; i < half; i++) {
		double ang = (double)pos * pow(theta, -2.0 * (double)i / (double)head_dim);
		double c = cos(ang), s = sin(ang);
		float x1 = x[i], x2 = x[i + half];
		x[i]        = (float)((double)x1 * c - (double)x2 * s);
		x[i + half] = (float)((double)x2 * c + (double)x1 * s);
	}
}

// Per-token symmetric int8 quant; returns max|x| (dequant = weight_scale*max|x|/127).
static inline double attn_act_quant_int8(const float *x, int n, int8_t *xq)
{
	double amax = 1e-5;
	for (int i = 0; i < n; i++) { double a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }
	double scale = 127.0 / amax;
	for (int i = 0; i < n; i++) {
		long r = lrint((double)x[i] * scale);
		if (r > 127) r = 127; else if (r < -128) r = -128;
		xq[i] = (int8_t)r;
	}
	return amax;
}

// One attention decode step at position `pos`. q_int/k_int/v_int: FPGA projection
// outputs for this token; amax_x: the input's per-token absmax (shared q/k/v).
// kcache/vcache: [n_kv_heads * max_seq * head_dim], persist across steps (k stored
// post-RoPE, v raw). Writes the post-attn_sub_norm vector to o_normed[hidden] and its
// int8 requant to o_q[hidden]; returns amax_o. scratch: >= n_heads*head_dim + max_seq.
static inline double attn_decode_step(const attn_cfg_t *c, int pos,
	const int32_t *q_int, const int32_t *k_int, const int32_t *v_int, double amax_x,
	float *kcache, float *vcache, float *o_normed, int8_t *o_q, float *scratch)
{
	const int H = c->head_dim, NH = c->n_heads, NKV = c->n_kv_heads;
	const int rep = NH / NKV, KVROW = c->max_seq * H;
	const double dqq = c->sq * amax_x / 127.0;
	const double dqk = c->sk * amax_x / 127.0;
	const double dqv = c->sv * amax_x / 127.0;
	float *q = scratch;                 // [NH*H]
	float *prob = scratch + NH * H;     // [max_seq]

	// dequant q (all heads), RoPE per head
	for (int h = 0; h < NH; h++) {
		float *qh = q + h * H;
		for (int d = 0; d < H; d++) qh[d] = (float)((double)q_int[h * H + d] * dqq);
		attn_rope(qh, pos, H, c->theta);
	}
	// dequant k,v (kv heads); RoPE k; append both to cache at `pos`
	for (int kv = 0; kv < NKV; kv++) {
		float *kdst = kcache + kv * KVROW + pos * H;
		float *vdst = vcache + kv * KVROW + pos * H;
		for (int d = 0; d < H; d++) {
			kdst[d] = (float)((double)k_int[kv * H + d] * dqk);
			vdst[d] = (float)((double)v_int[kv * H + d] * dqv);
		}
		attn_rope(kdst, pos, H, c->theta);
	}

	const double scaling = 1.0 / sqrt((double)H);
	for (int h = 0; h < NH; h++) {
		const float *qh = q + h * H;
		const float *kbase = kcache + (h / rep) * KVROW;
		const float *vbase = vcache + (h / rep) * KVROW;
		double mx = -1e30;
		for (int j = 0; j <= pos; j++) {                 // causal: 0..pos
			double dot = 0.0;
			const float *kj = kbase + j * H;
			for (int d = 0; d < H; d++) dot += (double)qh[d] * (double)kj[d];
			dot *= scaling;
			prob[j] = (float)dot;
			if (dot > mx) mx = dot;
		}
		double sum = 0.0;
		for (int j = 0; j <= pos; j++) { double e = exp((double)prob[j] - mx); prob[j] = (float)e; sum += e; }
		double inv = 1.0 / sum;
		float *oh = o_normed + h * H;                    // reuse o_normed as ctx scratch
		for (int d = 0; d < H; d++) {
			double acc = 0.0;
			for (int j = 0; j <= pos; j++) acc += (double)prob[j] * (double)vbase[j * H + d];
			oh[d] = (float)(acc * inv);
		}
	}

	// attn_sub_norm (RMSNorm over hidden) in place
	double ss = 0.0;
	for (int i = 0; i < c->hidden; i++) ss += (double)o_normed[i] * (double)o_normed[i];
	double rms = 1.0 / sqrt(ss / (double)c->hidden + c->eps);
	for (int i = 0; i < c->hidden; i++)
		o_normed[i] = (float)((double)o_normed[i] * rms * c->norm_w[i]);

	return attn_act_quant_int8(o_normed, c->hidden, o_q);
}

#endif /* ATTN_GLUE_H */
