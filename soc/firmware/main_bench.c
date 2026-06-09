// ternfpga on-board throughput harness: load resident weights + activation, pulse
// `run`, and read back the hardware-measured compute cycle count — the engine's
// true 1-tile/cycle rate on silicon, the basis for the energy/token number.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#include "testvec_bench.h"

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();

	printf("\n=== ternfpga throughput harness (K=%d NT=%d M=%d) ===\n", TV_K, TV_NT, TV_M);

	bench_nt_write(TV_NT);
	bench_m_rows_write(TV_M);

	for (int t = 0; t < TV_NT; t++) {                 // load activation (resident)
		bench_x_waddr_write(t);
		bench_x_wdata_write(X[t]);
		bench_x_we_write(1);
	}
	for (unsigned i = 0; i < sizeof(WT) / sizeof(WT[0]); i++) {   // load weights (resident)
		bench_wm_waddr_write(i);
		bench_wm_wdata_write(WT[i]);
		bench_wm_we_write(1);
	}

	bench_run_write(1);                               // TIMED: replay at 1 tile/cycle

	unsigned spins = 0;
	while ((bench_status_read() & 1u) == 0u) {
		if (++spins > 8000000u) { printf("TIMEOUT\n"); break; }
	}

	int ok = 1;
	for (int m = 0; m < TV_M; m++) {
		bench_rd_addr_write(m);
		int y = (int)bench_rd_data_read();
		if (y != EXP[m]) { ok = 0; printf("  y[%d] = %d  exp %d  BAD\n", m, y, EXP[m]); }
	}

	unsigned cyc = bench_cyc_read();
	unsigned tiles = (unsigned)TV_NT * (unsigned)TV_M;

	if (ok) printf("BENCH_ONBOARD_PASS  (%d rows bit-exact)\n", TV_M);
	else    printf("BENCH_ONBOARD_FAIL\n");
	printf("MEASURED  tiles=%u  cycles=%u  cyc_per_tile=%u.%02u  (K=%d @100MHz)\n",
	       tiles, cyc, cyc / tiles, (cyc % tiles) * 100u / tiles, TV_K);

	while (1) { }
	return 0;
}
