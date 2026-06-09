"""Probe the real microsoft/BitNet-b1.58-2B-4T ATTENTION block to pin down the exact
arithmetic our golden (models/attn_ref.py) + host glue must match: GQA head counts, the
sub-norm (if any) before o_proj, RoPE theta, head_dim, eps.

  python models/inspect_bitnet_attn.py            # prints to stdout (needs torch + model)
"""
import inspect
import importlib

import torch
from transformers import AutoModelForCausalLM

M = "microsoft/BitNet-b1.58-2B-4T"


def src(obj):
    try:
        return inspect.getsource(obj)
    except Exception as e:  # noqa: BLE001
        return f"<no source: {e}>"


def main():
    print(f"loading {M} (float32 cpu) ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    cfg = m.config
    for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "rope_theta", "rms_norm_eps", "max_position_embeddings", "num_hidden_layers",
              "vocab_size", "tie_word_embeddings"):
        print(f"  cfg.{k} = {getattr(cfg, k, None)}")

    layer = m.model.layers[0]
    print("\n=== decoder layer children (norm placement) ===")
    for n, c in layer.named_children():
        print(f"  {n}: {type(c).__name__}")

    attn = layer.self_attn
    print("\n=== self_attn ===")
    print("type:", type(attn).__name__, "module:", type(attn).__module__)
    for n, c in attn.named_children():
        extra = ""
        if hasattr(c, "weight") and getattr(c, "weight", None) is not None:
            extra = f" weight{tuple(c.weight.shape)} dtype={c.weight.dtype}"
        bufs = [bn for bn, _ in c.named_buffers()] if hasattr(c, "named_buffers") else []
        print(f"  {n}: {type(c).__name__}{extra} buffers={bufs}")
    # scalar attrs that matter
    for a in ("num_heads", "num_key_value_heads", "head_dim", "num_key_value_groups",
              "scaling", "rope_theta", "layer_idx"):
        if hasattr(attn, a):
            print(f"  attn.{a} = {getattr(attn, a)}")

    print("\n=== Attention.forward ===\n", src(type(attn).forward))

    # the sub-norm before o_proj (BitNet attn_sub_norm), if present
    for n, c in attn.named_modules():
        if "norm" in n.lower() and n:
            print(f"  sub-norm module: {n}: {type(c).__name__} "
                  f"weight={tuple(c.weight.shape) if hasattr(c, 'weight') else None}")

    qp = attn.q_proj
    print("\n=== q_proj type:", type(qp).__name__, "module:", type(qp).__module__)
    mod = importlib.import_module(type(qp).__module__)
    for name in dir(mod):
        if any(k in name.lower() for k in ("rotary", "rotate_half", "apply_rotary")):
            obj = getattr(mod, name)
            if callable(obj):
                print(f"\n=== {name} ===\n", src(obj))


if __name__ == "__main__":
    main()
