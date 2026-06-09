// On-board ffn_glue_unit: load gate/up/w, run, read h_q[f] + max|N| bit-exact vs the oracle,
// and read the hardware cycle count — the silicon-measured FFN-glue latency. Completes the FFN
// glue's PyTorch->sim->silicon chain.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec_ffnglue.h"

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);            // LiteX UART is IRQ-driven — required or the console stalls
#endif
	uart_init();
	printf("\n=== ternfpga ffn_glue_unit on silicon (F=%d) ===\n", TV_F);

	for (int f = 0; f < TV_F; f++) {
		ffng_waddr_write(f);
		ffng_gate_wdata_write((uint32_t)FG_GATE[f]);
		ffng_up_wdata_write((uint32_t)FG_UP[f]);
		ffng_w_wdata_write((uint16_t)FG_W[f]);
		ffng_we_write(1);
	}
	ffng_f_count_write(TV_F);
	printf("loaded; running...\n");

	// `done` is a 1-cycle pulse the slow CPU poll can miss; instead wait for cycle_count to
	// freeze (the FSM stops incrementing it on return to IDLE) — robust completion detection.
	ffng_start_write(1);
	unsigned cyc = 0, prev = 0xffffffffu, stable = 0;
	for (unsigned i = 0; i < 4000000u; i++) {
		cyc = ffng_cyc_read();
		if (cyc != 0u && cyc == prev) { if (++stable > 200u) break; } else stable = 0;
		prev = cyc;
	}
	uint32_t alo = ffng_amax_lo_read(), amid = ffng_amax_mid_read(), ahi = ffng_amax_hi_read();
	int bad = 0;
	for (int f = 0; f < TV_F; f++) {
		ffng_rd_addr_write(f);
		uint32_t raw = ffng_rd_data_read() & 0xffu;
		int got = (raw & 0x80u) ? (int)(raw | 0xffffff00u) : (int)raw;
		if (got != (int)FG_HQ[f]) { if (bad < 6) printf("  h_q[%d] = %d exp %d\n", f, got, (int)FG_HQ[f]); bad++; }
	}
	if (alo != TV_AMAX_LO || amid != TV_AMAX_MID || ahi != TV_AMAX_HI) {
		printf("  amaxN %08x_%08x_%08x != %08x_%08x_%08x\n", ahi, amid, alo,
		       (unsigned)TV_AMAX_HI, (unsigned)TV_AMAX_MID, (unsigned)TV_AMAX_LO);
		bad++;
	}

	if (bad == 0) printf("FFNGLUE_ONBOARD_PASS  (%d h_q + max|N| bit-exact)\n", TV_F);
	else          printf("FFNGLUE_ONBOARD_FAIL  (%d mismatches)\n", bad);
	printf("MEASURED ffn-glue cycles/layer=%u (F=%d) vs host 2.58M -> %ux\n", cyc, TV_F, 2580000u / (cyc ? cyc : 1));

	while (1) { }
	return 0;
}
