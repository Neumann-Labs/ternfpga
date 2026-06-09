// Host unit test for the attention glue (attn_glue.h) vs the NumPy golden (attn_ref).
// Runs the per-token decode steps with a growing KV cache; the float glue (RoPE + GQA +
// causal softmax + sub_norm) must reproduce the golden's pre-o_proj vector to float
// precision. Compile + run natively before it goes on-board:
//   python gen_attn_testvec.py > attn_testvec.h
//   cc -O2 -o test_attn_glue test_attn_glue.c -lm && ./test_attn_glue
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "attn_glue.h"
#include "attn_testvec.h"

int main(void)
{
	attn_cfg_t c = {
		.hidden = TV_HID, .n_heads = TV_NH, .n_kv_heads = TV_NKV, .head_dim = TV_HD,
		.max_seq = TV_MAXSEQ, .theta = TV_THETA, .eps = TV_EPS,
		.sq = TV_SQ, .sk = TV_SK, .sv = TV_SV, .so = TV_SO, .norm_w = NORM_W,
	};
	static float kcache[TV_NKV * TV_MAXSEQ * TV_HD];
	static float vcache[TV_NKV * TV_MAXSEQ * TV_HD];
	float o_normed[TV_HID];
	int8_t o_q[TV_HID];
	float scratch[TV_NH * TV_HD + TV_MAXSEQ];

	double max_abs = 0.0, max_rel = 0.0, num = 0.0, dg = 0.0, dr = 0.0;
	for (int t = 0; t < TV_T; t++) {
		attn_decode_step(&c, t,
			&Q_INT[t * TV_HID], &K_INT[t * TV_NKV * TV_HD], &V_INT[t * TV_NKV * TV_HD],
			AMAX_X[t], kcache, vcache, o_normed, o_q, scratch);
		for (int i = 0; i < TV_HID; i++) {
			double got = o_normed[i], exp = EXP_O[t * TV_HID + i];
			double e = fabs(got - exp);
			if (e > max_abs) max_abs = e;
			double r = e / (fabs(exp) + 1e-6);
			if (r > max_rel) max_rel = r;
			num += got * exp; dg += got * got; dr += exp * exp;
		}
	}
	double cos = num / (sqrt(dg) * sqrt(dr) + 1e-12);
	printf("tokens=%d hidden=%d heads=%d/%d d=%d\n", TV_T, TV_HID, TV_NH, TV_NKV, TV_HD);
	printf("max abs err %.3e   max rel err %.3e   cosine %.7f\n", max_abs, max_rel, cos);
	int ok = (cos > 0.99999) && (max_rel < 1e-2);
	printf(ok ? "ATTN_GLUE_C_PASS\n" : "ATTN_GLUE_C_FAIL\n");
	return ok ? 0 : 1;
}
