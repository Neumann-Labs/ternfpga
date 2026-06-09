// Measure the VexRiscv host-glue cost for ONE BitNet-2B decoder layer at real width.
// The FPGA matmul rate is already measured (1.00 cyc/tile, Phase 2, dimension-independent);
// the glue (soft-float RMSNorm / RoPE / softmax / requant on the soft CPU) is the unknown.
// Times each piece with the LiteX timer0 (free-running down-counter at sys_clk = 100 MHz).
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "fw_mathf.h"      // soft-float libm — this LiteX/picolibc build ships none
#include "attn_glue.h"
#include "ffn_glue.h"

#define HID 2560
#define INTER 6912
#define NH 20
#define NKV 5
#define HD 128
#define MAXSEQ 128
#define POS 64                 // representative decode position (attends to a 65-deep cache)

static int32_t q_int[HID], k_int[NKV * HD], v_int[NKV * HD];
static int32_t gate_int[INTER], up_int[INTER];
static double norm_w[HID], ffn_w[INTER];
static float kcache[NKV * MAXSEQ * HD], vcache[NKV * MAXSEQ * HD];
static float o_normed[HID], scratch[NH * HD + MAXSEQ], xbuf[HID];
static int8_t o_q[HID], hq[INTER];

static inline void timer_start(void)
{
	timer0_en_write(0);
	timer0_reload_write(0xffffffff);
	timer0_load_write(0xffffffff);
	timer0_en_write(1);
}

static inline uint32_t timer_now(void)
{
	timer0_update_value_write(1);
	return timer0_value_read();             // down-counter: elapsed = start - end
}

static void rms_norm_f(float *x, const double *w, int n, double eps)
{
	double ss = 0.0;
	for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
	double r = 1.0 / sqrt(ss / (double)n + eps);
	for (int i = 0; i < n; i++) x[i] = (float)((double)x[i] * r * w[i]);
}

int main(void)
{
	uart_init();
	timer_start();
	printf("\n=== ternfpga glue-cycle bench (hidden=%d inter=%d pos=%d @100MHz) ===\n",
	       HID, INTER, POS);

	for (int i = 0; i < HID; i++) {
		q_int[i] = (i * 37 % 251) - 125; norm_w[i] = 0.9 + 0.001 * (i % 50);
		xbuf[i] = 0.01f * ((i % 127) - 63);
	}
	for (int i = 0; i < NKV * HD; i++) { k_int[i] = (i * 53 % 251) - 125; v_int[i] = (i * 97 % 251) - 125; }
	for (int i = 0; i < INTER; i++) {
		gate_int[i] = (i * 29 % 401) - 200; up_int[i] = (i * 41 % 401) - 200;
		ffn_w[i] = 0.9 + 0.001 * (i % 50);
	}
	for (int i = 0; i < NKV * MAXSEQ * HD; i++) {
		kcache[i] = 0.01f * ((i % 127) - 63); vcache[i] = 0.01f * ((i % 131) - 65);
	}

	attn_cfg_t c = { .hidden = HID, .n_heads = NH, .n_kv_heads = NKV, .head_dim = HD,
		.max_seq = MAXSEQ, .theta = 500000.0, .eps = 1e-5,
		.sq = 1.2, .sk = 1.8, .sv = 2.3, .so = 0.96, .norm_w = norm_w };

	uint32_t t0, t1;
	t0 = timer_now(); rms_norm_f(xbuf, norm_w, HID, 1e-5);                 t1 = timer_now();
	uint32_t c_norm = t0 - t1;
	t0 = timer_now(); double ax = attn_act_quant_int8(xbuf, HID, o_q);     t1 = timer_now();
	uint32_t c_quant = t0 - t1;
	t0 = timer_now();
	attn_decode_step(&c, POS, q_int, k_int, v_int, ax, kcache, vcache, o_normed, o_q, scratch);
	t1 = timer_now();
	uint32_t c_attn = t0 - t1;
	double sh;
	t0 = timer_now(); ffn_glue_hq(gate_int, up_int, ffn_w, INTER, hq, &sh); t1 = timer_now();
	uint32_t c_ffn = t0 - t1;

	// per-layer glue: 2x RMSNorm + 2x input-quant (attn + ffn) + attn decode step + ffn glue
	uint32_t glue = 2u * c_norm + 2u * c_quant + c_attn + c_ffn;
	printf("GLUE_CYC  norm=%u quant=%u attn=%u ffn=%u  per_layer=%u\n",
	       c_norm, c_quant, c_attn, c_ffn, glue);
	printf("CHK o_q[3]=%d hq[7]=%d sumHsq=%g amax=%g\n", (int)o_q[3], (int)hq[7], sh, ax);

	while (1) { }
	return 0;
}
