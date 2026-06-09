// On-board engine+ffn_glue cooperation: the ternary engine computes the gate/up GEMVs, its int32
// outputs feed the ffn_glue unit (relu^2*up*w + int8 requant), h_q read back bit-exact — the first
// multi-accelerator end-to-end computation on the board (two custom accelerators co-resident on one
// 35T, cooperating on real silicon).
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec_coresident.h"

static void engine_gemv(const unsigned short *WT, int *y)   // run one ternary GEMV, read M rows
{
	eng_ctl_write(1);                                       // start: latch dims, reset
	for (unsigned i = 0; i < (unsigned)(TV_M * TV_NT); i++)
		eng_w_tile_write(WT[i]);                            // stream M*NT tiles, row-major
	unsigned spins = 0;
	while ((eng_status_read() & 1u) == 0u) if (++spins > 8000000u) break;
	for (int m = 0; m < TV_M; m++) { eng_rd_addr_write(m); y[m] = (int)eng_rd_data_read(); }
}

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();
	printf("\n=== ternfpga engine+ffn_glue co-resident (hidden=%d INTER=%d) ===\n", TV_K * TV_NT, TV_M);

	eng_nt_write(TV_NT);
	eng_m_rows_write(TV_M);
	for (int t = 0; t < TV_NT; t++) { eng_x_waddr_write(t); eng_x_wdata_write(X[t]); eng_x_we_write(1); }

	int gate_int[TV_M], up_int[TV_M];
	engine_gemv(WGT, gate_int);                             // gate = Wg @ x   (on the engine)
	engine_gemv(WUT, up_int);                               // up   = Wu @ x

	int gbad = 0;
	for (int m = 0; m < TV_M; m++) if (gate_int[m] != EXP_GATE[m]) gbad++;

	for (int m = 0; m < TV_M; m++) {                        // engine outputs -> ffn_glue inputs
		ffng_waddr_write(m);
		ffng_gate_wdata_write((uint32_t)gate_int[m]);
		ffng_up_wdata_write((uint32_t)up_int[m]);
		ffng_w_wdata_write((uint16_t)WQ[m]);
		ffng_we_write(1);
	}
	ffng_f_count_write(TV_M);
	ffng_start_write(1);
	unsigned cyc = 0, prev = 0xffffffffu, stable = 0;       // wait for cycle_count to freeze
	for (unsigned i = 0; i < 4000000u; i++) {
		cyc = ffng_cyc_read();
		if (cyc != 0u && cyc == prev) { if (++stable > 200u) break; } else stable = 0;
		prev = cyc;
	}

	int bad = 0;
	for (int m = 0; m < TV_M; m++) {
		ffng_rd_addr_write(m);
		uint32_t raw = ffng_rd_data_read() & 0xffu;
		int got = (raw & 0x80u) ? (int)(raw | 0xffffff00u) : (int)raw;
		if (got != (int)EXP_HQ[m]) { if (bad < 6) printf("  h_q[%d] = %d exp %d\n", m, got, (int)EXP_HQ[m]); bad++; }
	}
	uint32_t alo = ffng_amax_lo_read(), amid = ffng_amax_mid_read(), ahi = ffng_amax_hi_read();
	if (alo != TV_AMAX_LO || amid != TV_AMAX_MID || ahi != TV_AMAX_HI) bad++;

	printf("engine gate/up GEMV: %s (%d row mismatches)\n", gbad ? "FAIL" : "ok", gbad);
	if (bad == 0 && gbad == 0)
		printf("COMBINED_ONBOARD_PASS  (engine -> ffn_glue, %d h_q bit-exact, ffn-glue %u cyc)\n", TV_M, cyc);
	else
		printf("COMBINED_ONBOARD_FAIL  (%d glue + %d engine mismatches)\n", bad, gbad);

	while (1) { }
	return 0;
}
