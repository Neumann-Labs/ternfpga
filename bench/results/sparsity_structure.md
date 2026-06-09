# Activation-sparsity STRUCTURE — Risk-2 verdict: genuinely unstructured

Phase-3 #33. The deep-research review flagged Risk-2: ternfpga's differentiator is *per-token
**unstructured** activation sparsity* — but if the zero pattern were static or N:M-regular, a
structured router (TENET-style) or static pruning would capture it and the differentiator would
collapse. This probe (`models/sparsity_structure.py`) measures the **structure**, not just the
fraction, by hooking every `down_proj` input of `microsoft/BitNet-b1.58-2B-4T` (62 tokens, 30
layers, intermediate 6912).

## Measured (per-layer, summary)
```
MEAN  sparsity=59.5%  sometimes(data-dep)=93.9%  jaccard=0.421  N:M-capture=68.9%
VERDICT: UNSTRUCTURED (data-dependent — gather differentiator holds)
```
| metric | mean | meaning |
|---|---:|---|
| **zero fraction** | **59.5%** | matches the earlier 59.8% (`activation_sparsity.md`) — consistent |
| **always-zero** channels | ~4% | the only statically-prunable part (small) |
| **always-active** channels | ~0.5% | trivially few |
| **"sometimes" (data-dependent)** | **93.9%** | the active set is overwhelmingly chosen *per token* |
| **token↔token Jaccard** | **0.42** | two tokens share <½ their active channels — the pattern shifts |
| **static N:M capture** (50%/block-of-32) | **68.9%** | a fixed mask mispredicts ~31% of the zeros |

Per-layer: sparsity ranges 41–80% (lowest at layer 0, peaks ~80% in early layers); the
data-dependent fraction is **>93% in 28 of 30 layers** (only layers 1–2, the highest-static, dip
to 56–72%). The pattern is consistent across the whole stack.

## Verdict (honest)
**The sparsity is genuinely unstructured / data-dependent — the differentiator holds.**
- Only ~4% of channels are statically zero (capturable by weight pruning); **~94% are
  data-dependent**, decided per token by the squared-ReLU gate.
- A static **N:M router captures only ~69%** of the zeros — it would still fetch ~31% of the
  truly-zero columns (wasted bandwidth/energy) AND would have to over-provision. The on-fabric
  **variable-count gather** of the *actual* zeros (Phase-2 `down_proj` gather, 56% byte savings
  measured) captures what N:M cannot.
- Low Jaccard (0.42) confirms the active set is not a stable subset — static specialization can't
  substitute.

This is the mechanism the surrounding work does *not* occupy: TeLLMe v2 exploits **no** activation
sparsity; FlightLLM / ETH-ternaryLLM exploit **structured weight** sparsity; TENET uses **structured
N:M**. ternfpga's data-dependent unstructured gather is distinct — and this probe shows the BitNet
zeros genuinely *are* unstructured, so the gather is the right (and unoccupied) tool.

## Caveats
- One prompt / 62 tokens, BitNet-2B-4T (the validated model). A larger corpus would tighten the
  numbers but the per-layer consistency (28/30 layers >93% data-dependent) is unlikely to flip.
- The N:M-capture metric uses a globally-most-active static mask (a generous structured baseline);
  a smarter dynamic N:M (per-token TopK in hardware, like TENET) captures more but is itself a
  structured *approximation* of the unstructured gather and costs a routing network.

_Reproduce:_ `python models/sparsity_structure.py --device cuda` (gpu-venv loads the model).
