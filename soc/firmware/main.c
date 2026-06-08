// ternfpga on-board GEMV firmware: drive the ternary engine from the VexRiscv
// CPU over the CSR bus, read y back, and check it against the NumPy golden.
//
// Build: drop this (+ testvec.h from gen_testvec.py) into a LiteX bare-metal
// demo dir and `make`; load with `litex_term --kernel=demo.bin`. Replaces the
// stock demo main() with a one-shot self-test.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec.h"

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();

	printf("\n=== ternfpga on-board GEMV (K=%d, M=%d) ===\n", TV_K, TV_M);

	ternary_x_write(X);                  // activation vector: K int8 packed into x_flat
	ternary_ctl_write(1);                // start: latch x, reset counters
	for (unsigned b = 0; b < sizeof(WB); b++)
		ternary_wbyte_write(WB[b]);      // stream dense base-3 weight bytes (M*BPR)

	unsigned spins = 0;
	while ((ternary_status_read() & 1u) == 0u) {
		if (++spins > 4000000u) { printf("TIMEOUT waiting for done\n"); break; }
	}

	int ok = 1;
	for (int m = 0; m < TV_M; m++) {
		ternary_rd_addr_write(m);
		int y = (int)ternary_rd_data_read();
		if (y != EXP[m]) { ok = 0; printf("  y[%d] = %d  exp %d   BAD\n", m, y, EXP[m]); }
	}
	if (ok) printf("TERNARY_ONBOARD_PASS  (%d rows bit-exact vs golden)\n", TV_M);
	else    printf("TERNARY_ONBOARD_FAIL\n");

	while (1) { }   // halt
	return 0;
}
