"""Validate the NumPy FFN golden (ffn_ref.ffn_block) against the real PyTorch BitNet
FFN, on captured activations + extracted ternary weights. Confirms our integer
arithmetic matches the model *before* we build the RTL (TDD: the spec is the test).

  python models/validate_ffn.py --device cuda     # gpu-venv loads the model
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ffn_ref import ffn_block

MODEL = "microsoft/BitNet-b1.58-2B-4T"
PROMPT = "The future of efficient on-device AI inference is ternary weights and"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    load_kw = {} if args.device == "cpu" else {"device_map": args.device}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, **load_kw).eval()
    mlp = model.model.layers[args.layer].mlp

    cap = {}

    def hook(_m, inp, out):
        cap["x"] = inp[0].detach().float().cpu().numpy()[0]   # (seq, hidden)
        cap["y"] = out.detach().float().cpu().numpy()[0]

    h = mlp.register_forward_hook(hook)
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)
    h.remove()

    def ext(lin):
        W = lin.weight.detach().float().cpu().numpy()
        s = float(lin.weight_scale.detach().float().cpu().numpy().reshape(-1)[0])
        u = np.unique(W)
        assert set(np.round(u).tolist()).issubset({-1.0, 0.0, 1.0}), f"not ternary: {u[:8]}"
        return W, s

    Wg, sg = ext(mlp.gate_proj)
    Wu, su = ext(mlp.up_proj)
    Wd, sd = ext(mlp.down_proj)
    fn = mlp.ffn_sub_norm.weight.detach().float().cpu().numpy()
    eps = getattr(model.config, "rms_norm_eps", 1e-5)

    x, y_real = cap["x"], cap["y"]
    y_g, inter = ffn_block(x, Wg, sg, Wu, su, Wd, sd, fn, eps)

    err = np.abs(y_g - y_real)
    rel = err / (np.abs(y_real) + 1e-6)
    cos = float(np.dot(y_g.flatten(), y_real.flatten()) /
                (np.linalg.norm(y_g) * np.linalg.norm(y_real) + 1e-12))
    print(f"layer {args.layer}  x{x.shape}  y{y_real.shape}  weight_scales g={sg:.5f} u={su:.5f} d={sd:.5f}")
    print(f"gate sparsity (relu^2*up == 0): {np.mean(inter['h_pre_norm'] == 0)*100:.1f}%")
    print(f"max abs err  {err.max():.3e}   mean abs err {err.mean():.3e}")
    print(f"max rel err  {rel.max():.3e}   mean rel err {rel.mean():.3e}")
    print(f"cosine sim   {cos:.6f}")
    # int8 quant vs fp32 reference: expect tight agreement (the golden's int path is
    # actually *more* exact than the model's fp matmul, so residual is fp rounding).
    ok = cos > 0.999 and rel.mean() < 0.05
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
