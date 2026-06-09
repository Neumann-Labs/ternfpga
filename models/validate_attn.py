"""Validate the NumPy attention golden (attn_ref.attention) against the real PyTorch
BitNet attention, on captured activations + extracted ternary weights. Confirms our
RoPE / GQA / softmax / sub-norm + integer projections match the model *before* RTL.

  python models/validate_attn.py --device cuda     # gpu-venv loads the model
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_ref import attention

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
    attn = model.model.layers[args.layer].self_attn

    cap = {}

    def hook(_m, args, kwargs, out):
        x = args[0] if args else kwargs["hidden_states"]               # passed as kwarg
        cap["x"] = x.detach().float().cpu().numpy()[0]                 # (seq, hidden)
        y = out[0] if isinstance(out, (tuple, list)) else out          # forward returns a tuple
        cap["y"] = y.detach().float().cpu().numpy()[0]

    h = attn.register_forward_hook(hook, with_kwargs=True)
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

    Wq, sq = ext(attn.q_proj)
    Wk, sk = ext(attn.k_proj)
    Wv, sv = ext(attn.v_proj)
    Wo, so = ext(attn.o_proj)
    an = attn.attn_sub_norm.weight.detach().float().cpu().numpy()

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    theta = float(cfg.rope_theta)
    eps = float(getattr(cfg, "rms_norm_eps", 1e-5))

    x, y_real = cap["x"], cap["y"]
    y_g, inter = attention(x, Wq, sq, Wk, sk, Wv, sv, Wo, so, an,
                           n_heads, n_kv, head_dim, theta, eps)

    err = np.abs(y_g - y_real)
    rel = err / (np.abs(y_real) + 1e-6)
    cos = float(np.dot(y_g.flatten(), y_real.flatten()) /
                (np.linalg.norm(y_g) * np.linalg.norm(y_real) + 1e-12))
    print(f"layer {args.layer}  x{x.shape}  y{y_real.shape}  "
          f"heads {n_heads}/{n_kv} d={head_dim} theta={theta:g}")
    print(f"weight_scales q={sq:.5f} k={sk:.5f} v={sv:.5f} o={so:.5f}")
    print(f"max abs err  {err.max():.3e}   mean abs err {err.mean():.3e}")
    print(f"max rel err  {rel.max():.3e}   mean rel err {rel.mean():.3e}")
    print(f"cosine sim   {cos:.6f}")
    ok = cos > 0.999 and rel.mean() < 0.05
    print("ATTN_VALIDATE_PASS" if ok else "ATTN_VALIDATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
