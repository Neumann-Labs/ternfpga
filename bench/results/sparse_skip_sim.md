# Sparse-skip simulation results (Direction D lever)

Activation-sparse ternary GEMV gather engine (`rtl/ternary_gemv_sparse.sv`) that
fetches **only active rows** from weight memory. Measured in cocotb/Verilator
(`sim/tb_ternary_gemv_sparse.py`), `M=16`, `K=8`, ternary weights at 2 bits/weight
(dense = 32 bytes/GEMV). Regenerate: `make -C sim sparse`.

| sparsity label | active rows | rows fetched | weight bytes | bytes saved |
|---|---|---|---|---|
| dense   | 16/16 | 16 | 32/32 | 0.0% |
| 75%     | 12/16 | 12 | 24/32 | 25.0% |
| 50%     |  8/16 |  8 | 16/32 | 50.0% |
| 25%     |  4/16 |  4 |  8/32 | 75.0% |
| sparse1 |  1/16 |  1 |  2/32 | 93.8% |
| empty   |  0/16 |  0 |  0/32 | 100.0% |

**Result.** Weight-byte fetch scales **exactly** with activation density —
`rows_fetched == active rows` in every case, output bit-exact vs the NumPy
golden. Inactive neurons cost **zero memory traffic and zero compute cycles**.

**Why it matters.** At the 85–95% activation sparsity reported for relu-fied /
ProSparse FFNs (ProSparse Llama-2-7B = 89.3%, [arXiv:2402.13516](https://arxiv.org/abs/2402.13516)),
this engine fetches ~5–15% of the dense weight bytes. Because batch-1 decode is
DDR3-bandwidth-bound, that fetch reduction is a direct latency/energy win — and
it is *unstructured, per-token* sparsity, which a GPU's dense MAC array (or its
2:4-only sparse tensor cores) structurally cannot exploit.

**Scope.** This isolates the *fetch-skipping* lever; the weight memory is modeled
as a synchronous-read ROM. Measuring the actual sustained DDR3 bandwidth under
gathered (non-sequential) access on real silicon is a Phase-1 milestone — that
number determines how much of this simulated saving survives on the Arty.
