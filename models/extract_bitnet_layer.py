"""Extract a real ternary weight tile from a trained BitNet model.

Reads ONE weight tensor straight from safetensors (no model architecture, no
trust_remote_code), applies BitNet absmean ternarization, slices an MxK tile,
pairs it with a sample int8 activation, and saves a .npz the cocotb test loads.
Also reports the *real* weight sparsity (fraction of zeros) of the full layer —
the lever Direction D exploits.

  python models/extract_bitnet_layer.py --out models/data/real_tile.npz
"""
from __future__ import annotations
import argparse
import os

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from export_weights import ternarize_absmean, save_tile, sparsity

DEFAULT_REPO = "1bitLLM/bitnet_b1_58-large"   # 0.7B reproduction; single model.safetensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--weights-file", default="model.safetensors")
    ap.add_argument("--tensor", default=None, help="tensor name; default auto-picks an FFN proj")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", default="models/data/real_tile.npz")
    args = ap.parse_args()

    path = hf_hub_download(args.repo, args.weights_file)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        name = args.tensor
        if name is None:
            prefer = ("gate_proj.weight", "up_proj.weight", "down_proj.weight", "q_proj.weight")
            name = next((k for suf in prefer for k in keys if k.endswith(suf)), keys[0])
        W = f.get_tensor(name).float().cpu().numpy()

    print(f"repo={args.repo} tensor={name} shape={tuple(W.shape)} dtype=float")
    Wq, scale = ternarize_absmean(W)
    print(f"FULL-LAYER ternary sparsity (fraction zeros) = {sparsity(Wq):.3f}  (absmean scale={scale:.5f})")

    assert W.shape[0] >= args.M and W.shape[1] >= args.K, "tensor smaller than requested tile"
    tile = Wq[:args.M, :args.K]
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, size=args.K, dtype=np.int64)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_tile(args.out, tile, x)
    print(f"saved real {args.M}x{args.K} ternary tile -> {args.out}  (tile sparsity={sparsity(tile):.3f})")


if __name__ == "__main__":
    main()
