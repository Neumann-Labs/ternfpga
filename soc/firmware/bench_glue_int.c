// PURE-INTEGER glue-cycle bench for one BitNet-2B decoder layer (no float/double/libm —
// as lean as main_dmabw, which ran fine on-board where the soft-float glue trapped).
// Does the real per-layer glue op-structure with Q15 LUTs (RoPE cos/sin, softmax exp) and
// integer RMSNorm-into-requant (the cancellation trick), timed with timer0. Numerics are
// validated separately in models/glue_fixed_ref.py (cosine 0.999999); this measures cycles.
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "glue_luts.h"

#define HID 2560
#define INTER 6912
#define NH 20
#define NKV 5
#define HD 128
#define MAXSEQ 128
#define POS 64

static void uputs(const char *s) { while (*s) uart_write(*s++); }
static void uputu(uint32_t v)
{ char b[12]; int i = 12; b[--i] = 0; if (!v) b[--i] = '0';
  while (v) { b[--i] = (char)('0' + v % 10u); v /= 10u; } uputs(&b[i]); }
static void uline(const char *k, uint32_t v) { uputs(k); uputu(v); uputs("\n"); }

static void tstart(void)
{ timer0_en_write(0); timer0_reload_write(0); timer0_load_write(0xffffffff); timer0_en_write(1); }
static uint32_t tnow(void) { timer0_update_value_write(1); return timer0_value_read(); }

#define DRAM 0x46000000UL
static uint8_t *g_bump = (uint8_t *)DRAM;
static void *da(unsigned n) { void *p = g_bump; g_bump += (n + 15u) & ~15u; return p; }

static int32_t *q_int, *k_int, *v_int, *gate_int, *up_int, *qrot, *krot, *kcache, *vcache, *prob;
static int16_t *w_q15, *ffn_w;
static int8_t *xq8;

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);            // LiteX UART TX is IRQ-driven — without this the ring stalls ~8 chars in
#endif
	uart_init();
	tstart();
	uputs("\n=== ternfpga INTEGER glue bench (no float) hid=2560 inter=6912 pos=64 ===\n");

	q_int = da(4 * HID); k_int = da(4 * NKV * HD); v_int = da(4 * NKV * HD);
	gate_int = da(4 * INTER); up_int = da(4 * INTER);
	qrot = da(4 * NH * HD); krot = da(4 * NKV * HD);
	w_q15 = da(2 * HID); ffn_w = da(2 * INTER);
	kcache = da(4 * NKV * MAXSEQ * HD); vcache = da(4 * NKV * MAXSEQ * HD);
	prob = da(4 * MAXSEQ); xq8 = da(HID);

	for (int i = 0; i < HID; i++) { q_int[i] = (i * 37 % 4001) - 2000; w_q15[i] = (int16_t)(29491 + i % 2000); }
	for (int i = 0; i < NKV * HD; i++) { k_int[i] = (i * 53 % 4001) - 2000; v_int[i] = (i * 97 % 4001) - 2000; }
	for (int i = 0; i < INTER; i++) { gate_int[i] = (i * 29 % 4001) - 2000; up_int[i] = (i * 41 % 4001) - 2000;
		ffn_w[i] = (int16_t)(29491 + i % 2000); }
	for (int i = 0; i < NKV * MAXSEQ * HD; i++) { kcache[i] = (i * 7 % 2001) - 1000; vcache[i] = (i * 11 % 2001) - 1000; }
	uputs("setup ok (DRAM buffers)\n");

	uint32_t t0, t1; volatile int32_t sink = 0;

	// (1) RMSNorm -> int8 requant, integer cancellation (1/rms cancels), x2 (input_ln + post_ln)
	t0 = tnow();
	for (int pass = 0; pass < 2; pass++) {
		int32_t mx = 1;
		for (int i = 0; i < HID; i++) { int32_t h = (q_int[i] * (int32_t)w_q15[i]) >> 15;
			int32_t a = h < 0 ? -h : h; if (a > mx) mx = a; }
		for (int i = 0; i < HID; i++) { int32_t h = (q_int[i] * (int32_t)w_q15[i]) >> 15;
			int32_t r = (h * 127) / mx; xq8[i] = (int8_t)(r > 127 ? 127 : r < -128 ? -128 : r); }
		sink += xq8[3];
	}
	t1 = tnow(); uint32_t c_norm = t0 - t1;

	// (2) RoPE (Q15 LUT) on q (NH heads) + k (NKV heads) at POS
	t0 = tnow();
	{ const short *cs = &ROPE_COS[POS * LUT_HALF], *sn = &ROPE_SIN[POS * LUT_HALF];
	  for (int h = 0; h < NH; h++) { int32_t *o = &qrot[h * HD]; const int32_t *qi = &q_int[h * HD];
		for (int i = 0; i < LUT_HALF; i++) { int32_t a = qi[i], b = qi[i + LUT_HALF];
			o[i] = (a * cs[i] - b * sn[i]) >> LUT_Q; o[i + LUT_HALF] = (b * cs[i] + a * sn[i]) >> LUT_Q; } }
	  for (int h = 0; h < NKV; h++) { int32_t *o = &krot[h * HD]; const int32_t *ki = &k_int[h * HD];
		for (int i = 0; i < LUT_HALF; i++) { int32_t a = ki[i], b = ki[i + LUT_HALF];
			o[i] = (a * cs[i] - b * sn[i]) >> LUT_Q; o[i + LUT_HALF] = (b * cs[i] + a * sn[i]) >> LUT_Q; } } }
	t1 = tnow(); uint32_t c_rope = t0 - t1;

	// (3) scores (int64 dot) + softmax (exp LUT + integer normalize) + a@V, per q-head (GQA)
	t0 = tnow();
	{ int n_rep = NH / NKV;
	  for (int h = 0; h < NH; h++) { const int32_t *qh = &qrot[h * HD]; int kvh = h / n_rep;
		const int32_t *kc = &kcache[kvh * MAXSEQ * HD], *vc = &vcache[kvh * MAXSEQ * HD];
		int32_t mx = -2147483647;
		for (int j = 0; j <= POS; j++) { int64_t dot = 0; const int32_t *kj = &kc[j * HD];
			for (int d = 0; d < HD; d++) dot += (int64_t)qh[d] * kj[d];
			int32_t lg = (int32_t)(dot >> 12); prob[j] = lg; if (lg > mx) mx = lg; }
		int64_t sum = 0;
		for (int j = 0; j <= POS; j++) { int idx = (mx - prob[j]); if (idx < 0) idx = 0; if (idx >= EXP_N) idx = EXP_N - 1;
			int32_t e = EXP_LUT[idx]; prob[j] = e; sum += e; }
		if (!sum) sum = 1;
		for (int d = 0; d < HD; d++) { int64_t acc = 0;
			for (int j = 0; j <= POS; j++) acc += (int64_t)prob[j] * vc[j * HD + d];
			sink += (int32_t)(acc / sum); } } }
	t1 = tnow(); uint32_t c_attn = t0 - t1;

	// (4) FFN glue: h_q = round(relu(gate)^2 * up * w * 127 / max), PURE INTEGER (w as Q15)
	t0 = tnow();
	{ int64_t mxN = 1;
	  for (int f = 0; f < INTER; f++) { int32_t gp = gate_int[f] > 0 ? gate_int[f] : 0;
		int64_t H = (int64_t)gp * gp * up_int[f]; int64_t N = H * ffn_w[f];
		int64_t a = N < 0 ? -N : N; if (a > mxN) mxN = a; }
	  for (int f = 0; f < INTER; f++) { int32_t gp = gate_int[f] > 0 ? gate_int[f] : 0;
		int64_t H = (int64_t)gp * gp * up_int[f]; int64_t N = H * ffn_w[f];
		int32_t r = (int32_t)((N * 127) / mxN); sink += (int8_t)(r > 127 ? 127 : r < -128 ? -128 : r); } }
	t1 = tnow(); uint32_t c_ffn = t0 - t1;

	uint32_t glue = c_norm + c_rope + c_attn + c_ffn;
	uline("norm_x2=", c_norm);
	uline("rope=", c_rope);
	uline("attn(scores+softmax+av)=", c_attn);
	uline("ffn=", c_ffn);
	uputs("GLUE_INT_PER_LAYER="); uputu(glue); uputs(" sink="); uputu((uint32_t)(sink & 0xffff)); uputs("\n");
	uputs("DONE\n");

	while (1) { }
	return 0;
}
