// ternfpga on-board streaming-GEMV firmware: drive ternary_gemv_stream from the
// VexRiscv CPU over CSRs, read y back, and check it against the NumPy golden.
//
// Build: drop this (+ testvec_gemv.h from gen_testvec_gemv.py) into a LiteX
// bare-metal demo dir and `make`; load with litex_term serialboot. The scalable
// BRAM-centric GEMV computing on the fabric, read back over the bus.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec_gemv.h"

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();

	printf("\n=== ternfpga on-board streaming GEMV (K=%d NT=%d M=%d KT=%d) ===\n",
	       TV_K, TV_NT, TV_M, TV_K * TV_NT);

	gemv_nt_write(TV_NT);
	gemv_m_rows_write(TV_M);

	for (int t = 0; t < TV_NT; t++) {        // load activation into the engine BRAM
		gemv_x_waddr_write(t);
		gemv_x_wdata_write(X[t]);
		gemv_x_we_write(1);                  // commit x_mem[t] = X[t]
	}

	gemv_ctl_write(1);                       // start: latch dims, reset counters

	for (unsigned i = 0; i < sizeof(WT) / sizeof(WT[0]); i++)
		gemv_w_tile_write(WT[i]);            // stream M*NT weight tiles, row-major

	unsigned spins = 0;
	while ((gemv_status_read() & 1u) == 0u) {
		if (++spins > 8000000u) { printf("TIMEOUT waiting for done\n"); break; }
	}

	int ok = 1;
	for (int m = 0; m < TV_M; m++) {
		gemv_rd_addr_write(m);
		int y = (int)gemv_rd_data_read();
		if (y != EXP[m]) { ok = 0; printf("  y[%d] = %d  exp %d   BAD\n", m, y, EXP[m]); }
	}
	if (ok) printf("GEMV_ONBOARD_PASS  (%d rows bit-exact vs golden)\n", TV_M);
	else    printf("GEMV_ONBOARD_FAIL\n");

	while (1) { }   // halt
	return 0;
}
