# Models — ternary weight export pipeline

Bridges a trained BitNet b1.58 model to the RTL: extract a layer's weights,
absmean-ternarize them, and pack to the exact 2-bit encoding the hardware uses.

- **`ternary_ref.py`** — the encoding + golden, shared by the RTL tests and the
  export (one source of truth): int8 activations little-endian; ternary weights
  as 2-bit codes (`+1=01`, `-1=10`, `0=00`).
- **`export_weights.py`** — `ternarize_absmean(W)` (BitNet b1.58 weight quant),
  `pack_rows`, `save_tile`/`load_tile`, `sparsity`. Self-test: `python models/export_weights.py`.
- **`extract_bitnet_layer.py`** — read one weight tensor from a BitNet model via
  safetensors (no model arch / `trust_remote_code`), ternarize, slice an `M×K`
  tile, save `.npz`. Reports the real layer's weight-sparsity.
- **`data/real_tile.npz`** — a committed 16×8 tile from `1bitLLM/bitnet_b1_58-large`
  layer-0 `gate_proj` (test fixture; regenerate with the extractor).

```bash
python models/export_weights.py                                   # encoding round-trip self-test
python models/extract_bitnet_layer.py --out models/data/real_tile.npz
make -C sim real                                                  # real tile through the RTL, bit-exact
```

Measured (1bitLLM/bitnet_b1_58-large, layer-0 `gate_proj`, 4096×1536): **34% weight
sparsity** after absmean ternarization. Note this is *static weight* sparsity; the
larger lever Direction D targets is *activation* sparsity (85–95%, dynamic
per-token) — a separate runtime quantity. Both are exploitable by the gather
engine (`rtl/ternary_gemv_sparse.sv`).
