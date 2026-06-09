# down_proj activation-sparse gather (Direction D, column-sparse)

`down_proj` is `y[h] = sum_f Wd[h,f]·hq[f]`, and the activation `hq` is ~60% zero per token
(measured, [`activation_sparsity.md`](activation_sparsity.md)). The zero terms contribute nothing,
so we **compact** `hq` to its nonzero entries and **gather** only the matching `Wd` columns, then
run the *existing* dense streaming GEMV on the shorter vector — bit-exact, fetching only `active/F`
of the weight bytes.

This is the **column-sparse** complement to `ternary_gemv_sparse`'s **row-sparse** gather (which
skips `up_proj` output rows where `gate≤0`). Together they exploit the same per-token active set
the FFN produces.

## Verification (cocotb, K=16) — `sim/tb_gemv_gather.py`, F=256 contraction, H=24 outputs

| activation density | active / F | weight-tiles fetched | bytes saved | result |
|---|---|---|---|---|
| 100% (dense) | 256/256 | 384/384 | 0.0% | bit-exact |
| 60% | 151/256 | 240/384 | 37.5% | bit-exact |
| **40.2% (BitNet measured)** | 104/256 | 168/384 | **56.2%** | bit-exact |
| 15% (relu-fied) | 33/256 | 72/384 | **81.2%** | bit-exact |

Weight-byte fetch scales with activation density; the output is **bit-exact vs the dense golden**
at every density. (Savings track `1-density` minus a small K-tile padding overhead.)

## Why it matters
Batch-1 decode is DDR3-bandwidth-bound ([`scaling-feasibility.md`](../../docs/research/scaling-feasibility.md) §2),
so *not fetching* the zero-activation columns is a direct latency/energy win. It is **per-token,
unstructured** sparsity — a GPU's dense MAC array (or its 2:4-only sparse tensor cores)
structurally cannot exploit it. The **engine is unchanged**: the gather is a feed/DMA concern (the
DMA fetches only the gathered columns). At the measured BitNet b1.58 ~40% active, `down_proj`
fetches ~44% of the dense bytes; relu-fication toward ~10–15% active would push that to ~81–89%
saved (#25).

## Scope / next
This proves the gather is correct and quantifies the savings on the real engine. The hardware
**index-compaction** (`hq` → nonzero index list) + **DMA gather** (fetch only those DDR3 columns)
is task #24; this isolates the column-sparse fetch-reduction lever. Figure:
[`../plots/gather_savings.png`](../plots/gather_savings.png). Reproduce: `make -C sim gather`.
