// Host unit test for the FFN glue (ffn_glue.h) vs the NumPy golden (ffn_glue_ref).
// Compile + run on any x86/host before it goes on-board:
//   python gen_glue_testvec.py > glue_testvec.h
//   cc -O2 -o test_ffn_glue test_ffn_glue.c -lm && ./test_ffn_glue
#include <stdio.h>
#include <stdint.h>

#include "ffn_glue.h"
#include "glue_testvec.h"   // GATE_INT[TV_F], UP_INT[TV_F], NORM_W[TV_F], EXP_HQ[TV_F]

int main(void)
{
	int8_t hq[TV_F];
	double sumHsq, amaxN;

	amaxN = ffn_glue_hq(GATE_INT, UP_INT, NORM_W, TV_F, hq, &sumHsq);

	int bad = 0;
	for (int f = 0; f < TV_F; f++) {
		if (hq[f] != EXP_HQ[f]) {
			if (bad < 8) printf("  hq[%d] = %d  exp %d\n", f, hq[f], EXP_HQ[f]);
			bad++;
		}
	}
	if (bad == 0)
		printf("FFN_GLUE_C_PASS  (%d channels bit-exact vs ffn_glue_ref; max|N|=%.3g)\n",
		       TV_F, amaxN);
	else
		printf("FFN_GLUE_C_FAIL  (%d/%d channels mismatch)\n", bad, TV_F);

	return bad ? 1 : 0;
}
