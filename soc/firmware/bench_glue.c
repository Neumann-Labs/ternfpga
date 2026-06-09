// Measure the VexRiscv host-glue cost for ONE BitNet-2B decoder layer at real width.
// The FPGA matmul rate is already measured (1.00 cyc/tile, Phase 2, dimension-independent);
// the glue (soft-float RMSNorm / RoPE / softmax / requant on the soft CPU) is the unknown.
//
// I/O uses uart_write directly (a manual integer formatter), not printf's %d — the heavy
// soft-float binary makes picolibc's float-printf path trap on %d here. Timing uses the LiteX
// timer0 (down-counter @ sys_clk = 100 MHz), set up exactly as the BIOS does for Memspeed.
#include <stdint.h>

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

// ---- bare-metal I/O without printf %d ----------------------------------------------------
static void uputs(const char *s) { while (*s) uart_write(*s++); }
static void uputu(uint32_t v)
{
	char b[12]; int i = 12; b[--i] = 0;
	if (!v) b[--i] = '0';
	while (v) { b[--i] = (char)('0' + (v % 10u)); v /= 10u; }
	uputs(&b[i]);
}
static void uline(const char *k, uint32_t v) { uputs(k); uputu(v); uputs("\n"); }

// ---- timing via LiteX timer0 (down-counter, BIOS pattern) --------------------------------
static void timer_start(void)
{
	timer0_en_write(0);
	timer0_reload_write(0);
	timer0_load_write(0xffffffff);
	timer0_en_write(1);
}
static uint32_t timer_now(void)            // counts DOWN: elapsed = start - end
{
	timer0_update_value_write(1);
	return timer0_value_read();
}

// ---- big buffers in DDR3 (the ~0.8 MB KV cache overflows the small sram .bss) -------------
#define DRAM_SCRATCH 0x46000000UL
static uint8_t *g_bump = (uint8_t *)DRAM_SCRATCH;
static void *dalloc(unsigned n) { void *p = g_bump; g_bump += (n + 15u) & ~15u; return p; }

static int32_t *q_int, *k_int, *v_int, *gate_int, *up_int;
static double *norm_w, *ffn_w;
static float *kcache, *vcache, *o_normed, *scratch, *xbuf;
static int8_t *o_q, *hq;

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
	uputs("\n=== ternfpga glue-cycle bench hid="); uputu(HID);
	uputs(" inter="); uputu(INTER); uputs(" pos="); uputu(POS); uputs(" @100MHz ===\n");

	q_int = dalloc(sizeof(int32_t) * HID);
	k_int = dalloc(sizeof(int32_t) * NKV * HD);
	v_int = dalloc(sizeof(int32_t) * NKV * HD);
	gate_int = dalloc(sizeof(int32_t) * INTER);
	up_int = dalloc(sizeof(int32_t) * INTER);
	norm_w = dalloc(sizeof(double) * HID);
	ffn_w = dalloc(sizeof(double) * INTER);
	kcache = dalloc(sizeof(float) * NKV * MAXSEQ * HD);
	vcache = dalloc(sizeof(float) * NKV * MAXSEQ * HD);
	o_normed = dalloc(sizeof(float) * HID);
	scratch = dalloc(sizeof(float) * (NH * HD + MAXSEQ));
	xbuf = dalloc(sizeof(float) * HID);
	o_q = dalloc(HID);
	hq = dalloc(INTER);

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
	uputs("setup ok (buffers in DRAM)\n");

	attn_cfg_t c = { .hidden = HID, .n_heads = NH, .n_kv_heads = NKV, .head_dim = HD,
		.max_seq = MAXSEQ, .theta = 500000.0, .eps = 1e-5,
		.sq = 1.2, .sk = 1.8, .sv = 2.3, .so = 0.96, .norm_w = norm_w };

	uint32_t t0, t1;
	t0 = timer_now(); rms_norm_f(xbuf, norm_w, HID, 1e-5);                 t1 = timer_now();
	uint32_t c_norm = t0 - t1;  uline("norm=", c_norm);
	t0 = timer_now(); double ax = attn_act_quant_int8(xbuf, HID, o_q);     t1 = timer_now();
	uint32_t c_quant = t0 - t1; uline("quant=", c_quant);
	t0 = timer_now();
	attn_decode_step(&c, POS, q_int, k_int, v_int, ax, kcache, vcache, o_normed, o_q, scratch);
	t1 = timer_now();
	uint32_t c_attn = t0 - t1;  uline("attn=", c_attn);
	double sh;
	t0 = timer_now(); ffn_glue_hq(gate_int, up_int, ffn_w, INTER, hq, &sh); t1 = timer_now();
	uint32_t c_ffn = t0 - t1;   uline("ffn=", c_ffn);

	// per-layer glue: 2x RMSNorm + 2x input-quant (attn + ffn) + attn decode step + ffn glue
	uint32_t glue = 2u * c_norm + 2u * c_quant + c_attn + c_ffn;
	uputs("GLUE_CYC_PER_LAYER="); uputu(glue);
	uputs(" chk="); uputu((uint32_t)(o_q[3] + hq[7] + 256)); uputs("\n");
	uputs("DONE\n");

	while (1) { }
	return 0;
}
