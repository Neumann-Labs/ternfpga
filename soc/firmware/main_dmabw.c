// Measure the sustained DDR3 read bandwidth (the roofline) with a hardware DMA + cycle
// counter, then derive the full-model tok/s ceiling. Batch-1 decode streams every weight
// once per token, so this single GB/s number bounds tokens/sec — the Risk-1 test.
#include <stdio.h>
#include <stdint.h>

#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>

#define MODEL_MB_PER_TOK 175u   // BitNet ~0.7B ternary, 2-bit packed (~175 MB / token)
#define SOC_POWER_MW     489u   // Vivado-estimated SoC on-chip power (labeled estimate)

int main(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();

	printf("\n=== ternfpga DDR3 read-bandwidth roofline ===\n");

	unsigned dwb    = dmabw_dwbytes_read();        // native word width in bytes
	unsigned length = 2000000u;                    // 2e6 native words
	printf("native word = %u bytes; reading %u words = %u MiB\n",
	       dwb, length, (unsigned)((uint64_t)length * dwb / 1048576u));

	dmabw_base_write(0);
	dmabw_length_write(length);
	dmabw_start_write(1);

	unsigned spins = 0;
	while ((dmabw_done_read() & 1u) == 0u) {
		if (++spins > 300000000u) { printf("TIMEOUT (done never asserted)\n"); break; }
	}

	unsigned cycles = dmabw_cycles_read();
	unsigned words  = dmabw_words_read();
	unsigned chk    = dmabw_chksum_read();
	uint64_t bytes  = (uint64_t)words * dwb;

	// 100 MHz: bytes/s = bytes * 1e8 / cycles ; MB/s (1e6) = bytes * 100 / cycles
	unsigned bw_mbps = (unsigned)((bytes * 100ULL) / (uint64_t)cycles);
	// tok/s ceiling (x100 for 2 decimals) = bw_mbps / MODEL_MB_PER_TOK
	unsigned tok_x100 = bw_mbps * 100u / MODEL_MB_PER_TOK;
	// J/token floor (mJ) = power_mW / tok_s = power_mW * 100 / tok_x100
	unsigned mjtok = (tok_x100 ? SOC_POWER_MW * 100u / tok_x100 : 0u);

	printf("words=%u cycles=%u chksum=0x%08x\n", words, cycles, chk);
	printf("DDR3_READ_BW  bytes=%u cycles=%u  bw=%u MB/s\n",
	       (unsigned)bytes, cycles, bw_mbps);
	printf("ROOFLINE  model=%u MB/tok  tok_s_ceiling=%u.%02u  Jtok_floor~=%u mJ (est %u mW SoC)\n",
	       MODEL_MB_PER_TOK, tok_x100 / 100u, tok_x100 % 100u, mjtok, SOC_POWER_MW);

	while (1) { }
	return 0;
}
