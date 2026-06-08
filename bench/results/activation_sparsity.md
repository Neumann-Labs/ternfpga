# BitNet b1.58 FFN Activation Sparsity (measured)

**Model:** `microsoft/BitNet-b1.58-2B-4T` · torch.bfloat16 cuda forward · 6 diverse passages · `|x| < 1e-08` counts as zero.

Measures the per-token zero fraction of the vector feeding each FFN `down_proj` — i.e. the squared-ReLU-gated activation `ReLU(gate_proj(x))^2 . up_proj(x)`. Every zero is a `down_proj` weight column that need not be fetched or multiplied (Direction D). This is the figure our deep-research sweep could not find published for b1.58 specifically (`docs/research/scaling-feasibility.md`, angle 4).

## Headline

- **Overall activation sparsity: 59.8%**  (mean per-layer 59.8%, range 41.6%–78.9%).
- **Active units per token:** mean **2778**, p95 **4116**, max **6153** of **6912** → sizes the hardware gather buffer / index queue.
- Sample size: 12,480 (token × layer) activation vectors, 86,261,760 scalar activations.

## Per-layer

| layer | activation sparsity | mean active units / token |
|------:|--------------------:|--------------------------:|
| 0 | 44.2% | 3857 |
| 1 | 73.4% | 1842 |
| 2 | 78.9% | 1458 |
| 3 | 70.3% | 2052 |
| 4 | 66.1% | 2345 |
| 5 | 66.6% | 2308 |
| 6 | 69.9% | 2078 |
| 7 | 67.4% | 2255 |
| 8 | 61.4% | 2670 |
| 9 | 55.5% | 3075 |
| 10 | 51.4% | 3356 |
| 11 | 46.5% | 3701 |
| 12 | 45.7% | 3754 |
| 13 | 46.1% | 3724 |
| 14 | 43.7% | 3890 |
| 15 | 41.6% | 4037 |
| 16 | 47.2% | 3650 |
| 17 | 52.7% | 3271 |
| 18 | 55.2% | 3094 |
| 19 | 58.0% | 2906 |
| 20 | 62.8% | 2571 |
| 21 | 67.6% | 2240 |
| 22 | 63.9% | 2497 |
| 23 | 67.4% | 2251 |
| 24 | 67.8% | 2228 |
| 25 | 66.5% | 2314 |
| 26 | 66.7% | 2299 |
| 27 | 65.4% | 2390 |
| 28 | 64.3% | 2466 |
| 29 | 59.9% | 2775 |

## What it means for the build

- **Measured 60% sparsity (40% active) — NOT the 85–95% Direction D assumed.** BitNet b1.58's squared-ReLU gate zeros ~60% of `down_proj` inputs per token, so an activation-gather path fetches/multiplies only ~40% of `down_proj`'s weight columns — a real ~2.5× cut in `down_proj` traffic and FLOPs that a GPU cannot exploit (it does only rigid 2:4 structured sparsity). But it is **not** the 10–20× the README's Direction D claimed; that figure came from separate relu-fication / ProSparse work, not stock b1.58.
- **Sparsity is uneven by depth** — early layers (1–8) are ~65–79% sparse (skip most), middle layers (~11–16) only ~42–47% (little to skip). A static gather sized for the worst case (p95 active ≈ 4116 of 6912) wastes little.
- **Path to higher sparsity:** relu-fication / ProSparse fine-tuning reaches 85–95% FFN sparsity in the literature — a future lever if Direction D's payoff must grow. Until then, claim only the measured ~60%.
- **Gather cost:** the index queue need only hold ~p95 active entries (~4116), a modest BRAM cost; the open question is whether gather control is cheaper than streaming all 6912 columns dense at 40% density.

_Reproduce:_ `python models/measure_activation_sparsity.py --device cuda`
