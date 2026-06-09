// On-board attention_unit: load q/KV/LUT, run, read num[d]+sum_e bit-exact vs the oracle,
// and read the hardware cycle count — the silicon-measured attention latency. Completes
// attention's PyTorch->sim->silicon chain.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec_attn.h"

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);            // LiteX UART is IRQ-driven — required or the console stalls
#endif
	uart_init();
	printf("\n=== ternfpga attention_unit on silicon (D=%d T=%d) ===\n", TV_D, TV_T);

	for (int i = 0; i < TV_EXPN; i++) {           // exp LUT
		attn_lut_waddr_write(i); attn_lut_wdata_write(ATV_ELUT[i]); attn_lut_we_write(1);
	}
	for (int d = 0; d < TV_D; d++) {              // q
		attn_q_waddr_write(d); attn_q_wdata_write((uint16_t)ATV_Q[d]); attn_q_we_write(1);
	}
	for (int a = 0; a < TV_T * TV_D; a++) {        // KV cache (k+v together)
		attn_kv_waddr_write(a);
		attn_k_wdata_write((uint16_t)ATV_K[a]); attn_v_wdata_write((uint16_t)ATV_V[a]);
		attn_kv_we_write(1);
	}
	attn_t_keys_write(TV_T);
	attn_score_shift_write(TV_SHIFT);
	printf("loaded; running...\n");

	attn_start_write(1);
	unsigned spins = 0;
	while ((attn_status_read() & 1u) == 0u) { if (++spins > 20000000u) { printf("TIMEOUT\n"); break; } }

	unsigned cyc = attn_cyc_read();
	unsigned sum = attn_sum_e_read();
	int bad = 0;
	for (int d = 0; d < TV_D; d++) {
		attn_rd_addr_write(d);
		uint64_t raw = attn_rd_data_read();        // 48-bit
		int64_t got = (raw & (1ULL << 47)) ? (int64_t)(raw | ~((1ULL << 48) - 1)) : (int64_t)raw;
		if (got != ATV_NUM[d]) { if (bad < 6) printf("  num[%d] bad\n", d); bad++; }
	}
	if (sum != TV_SUM) { printf("  sum_e %u != %u\n", sum, (unsigned)TV_SUM); bad++; }

	if (bad == 0) printf("ATTN_ONBOARD_PASS  (%d num + sum_e bit-exact)\n", TV_D);
	else          printf("ATTN_ONBOARD_FAIL  (%d mismatches)\n", bad);
	printf("MEASURED attention cycles/query=%u (T=%d D=%d) sum_e=%u\n", cyc, TV_T, TV_D, sum);
	printf("  -> ~/layer(x20 heads)=%u vs host 16.2M\n", cyc * 20u);

	while (1) { }
	return 0;
}
