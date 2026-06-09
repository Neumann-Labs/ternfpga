// Integer-only FFN glue for the host (VexRiscv) — mirrors models/ffn_glue_ref.py.
//
// The FPGA computes the three int8×ternary→int32 matmuls (gate/up/down); the host
// does the glue between them. The key identity (ffn_glue_ref.py): the down_proj
// int8 input depends ONLY on the integer N[f] = relu(gate_int[f])^2 * up_int[f] *
// w[f] — every per-token dequant scale AND the RMSNorm normalizer cancel in the
// final absmax requant. So this needs only int64 (for the exact H) + double (for
// N), no float dequant / no RMSNorm divide:
//
//     h_q[f] = round( N[f] * 127 / max_g|N[g]| )
//
// (Compute-heavy on a soft CPU — an on-chip glue unit is the documented
// optimization; here we keep the validated host-split path.)
#ifndef FFN_GLUE_H
#define FFN_GLUE_H
#include <stdint.h>
#include <math.h>

// Per-token symmetric int8 quant of n float activations; returns max|x| (the
// dequant uses weight_scale * max|x| / 127). Mirrors ffn_ref.act_quant_int8.
static inline double ffn_act_quant_int8(const float *x, int n, int8_t *xq)
{
	double amax = 1e-5;
	for (int i = 0; i < n; i++) { double a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }
	double scale = 127.0 / amax;
	for (int i = 0; i < n; i++) {
		long r = lrint((double)x[i] * scale);     // ties-to-even, matches numpy round
		if (r > 127) r = 127; else if (r < -128) r = -128;
		xq[i] = (int8_t)r;
	}
	return amax;
}

// Integer-only glue: gate_int/up_int (int32, the FPGA GEMV outputs), w (per-channel
// fixed-point norm weight as double), F channels -> hq (int8 down_proj input).
// Two passes (find max|N|, then requant), recomputing H to avoid a wide N buffer.
// Returns max|N|; writes sum(H^2) for the host's final output dequant if requested.
// H = relu(gate)^2 * up fits int64 (|gate|,|up| ~ K*127 ~ 2^18 -> H ~ 2^55 < 2^63);
// casting that exact int64 to double rounds identically to numpy's object->float64.
static inline double ffn_glue_hq(const int32_t *gate_int, const int32_t *up_int,
                                 const double *w, int F, int8_t *hq, double *sumHsq_out)
{
	double amax = 1e-9, sumHsq = 0.0;
	for (int f = 0; f < F; f++) {
		int64_t g = gate_int[f] > 0 ? (int64_t)gate_int[f] : 0;
		int64_t H = g * g * (int64_t)up_int[f];
		double N = (double)H * w[f];
		double a = N < 0 ? -N : N;
		if (a > amax) amax = a;
		sumHsq += (double)H * (double)H;
	}
	double scale = 127.0 / amax;
	for (int f = 0; f < F; f++) {
		int64_t g = gate_int[f] > 0 ? (int64_t)gate_int[f] : 0;
		int64_t H = g * g * (int64_t)up_int[f];
		double N = (double)H * w[f];
		long r = lrint(N * scale);
		if (r > 127) r = 127; else if (r < -128) r = -128;
		hq[f] = (int8_t)r;
	}
	if (sumHsq_out) *sumHsq_out = sumHsq;
	return amax;
}

#endif /* FFN_GLUE_H */
