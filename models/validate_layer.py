"""Validate the NumPy decoder-layer golden (layer_ref.decoder_layer) against a real
BitNet decoder layer — the whole layer (2 norms + attention + FFN + residuals) end to
end, on captured I/O + extracted ternary weights.

  python models/validate_layer.py --device cuda
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from layer_ref import decoder_layer

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
    layer = model.model.layers[args.layer]

    cap = {}

    def hook(_m, a, kw, out):
        x = a[0] if a else kw["hidden_states"]
        cap["x"] = x.detach().float().cpu().numpy()[0]
        y = out[0] if isinstance(out, (tuple, list)) else out
        cap["y"] = y.detach().float().cpu().numpy()[0]

    h = layer.register_forward_hook(hook, with_kwargs=True)
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)
    h.remove()

    def ext(lin):
        W = lin.weight.detach().float().cpu().numpy()
        s = float(lin.weight_scale.detach().float().cpu().numpy().reshape(-1)[0])
        return W, s

    at, mlp = layer.self_attn, layer.mlp
    Wq, sq = ext(at.q_proj); Wk, sk = ext(at.k_proj); Wv, sv = ext(at.v_proj); Wo, so = ext(at.o_proj)
    Wg, sg = ext(mlp.gate_proj); Wu, su = ext(mlp.up_proj); Wd, sd = ext(mlp.down_proj)
    w = {
        "input_ln": layer.input_layernorm.weight.detach().float().cpu().numpy(),
        "post_ln": layer.post_attention_layernorm.weight.detach().float().cpu().numpy(),
        "attn_norm": at.attn_sub_norm.weight.detach().float().cpu().numpy(),
        "ffn_norm": mlp.ffn_sub_norm.weight.detach().float().cpu().numpy(),
        "Wq": Wq, "sq": sq, "Wk": Wk, "sk": sk, "Wv": Wv, "sv": sv, "Wo": Wo, "so": so,
        "Wg": Wg, "sg": sg, "Wu": Wu, "su": su, "Wd": Wd, "sd": sd,
    }
    cfg = model.config
    c = {"n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads,
         "head_dim": cfg.hidden_size // cfg.num_attention_heads,
         "theta": float(cfg.rope_theta), "eps": float(getattr(cfg, "rms_norm_eps", 1e-5))}

    x, y_real = cap["x"], cap["y"]
    y_g, _ = decoder_layer(x, w, c)

    err = np.abs(y_g - y_real)
    rel = err / (np.abs(y_real) + 1e-6)
    cos = float(np.dot(y_g.flatten(), y_real.flatten()) /
                (np.linalg.norm(y_g) * np.linalg.norm(y_real) + 1e-12))
    print(f"layer {args.layer}  x{x.shape}  y{y_real.shape}")
    print(f"max abs err {err.max():.3e}  mean abs err {err.mean():.3e}")
    print(f"max rel err {rel.max():.3e}  mean rel err {rel.mean():.3e}")
    print(f"cosine sim  {cos:.6f}")
    ok = cos > 0.999 and rel.mean() < 0.05
    print("LAYER_VALIDATE_PASS" if ok else "LAYER_VALIDATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
