"""Probe the real microsoft/BitNet-b1.58-2B-4T to pin down the exact FFN + BitLinear
arithmetic our golden (models/ffn_ref.py) and RTL must match. Dumps module structure,
the MLP/BitLinear forward source, and the activation/weight quant helpers.

  python models/inspect_bitnet_ffn.py            # prints to stdout
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
    print("hidden_size:", cfg.hidden_size, "intermediate_size:", cfg.intermediate_size)
    print("hidden_act:", getattr(cfg, "hidden_act", None))

    layer = m.model.layers[0]
    print("\n=== decoder layer children (where do the norms live?) ===")
    for n, c in layer.named_children():
        print(f"  {n}: {type(c).__name__}")

    mlp = layer.mlp
    print("\n=== MLP ===")
    print("type:", type(mlp).__name__)
    for n, c in mlp.named_children():
        extra = ""
        if hasattr(c, "weight"):
            extra = f" weight{tuple(c.weight.shape)} dtype={c.weight.dtype}"
        bufs = [bn for bn, _ in c.named_buffers()] if hasattr(c, "named_buffers") else []
        print(f"  {n}: {type(c).__name__}{extra} buffers={bufs}")
    print("act_fn:", getattr(mlp, "act_fn", None))
    print("\n=== MLP.forward ===\n", src(type(mlp).forward))

    gp = mlp.gate_proj
    print("=== gate_proj type:", type(gp).__name__, "module:", type(gp).__module__)
    print("\n=== BitLinear.forward ===\n", src(type(gp).forward))

    mod = importlib.import_module(type(gp).__module__)
    for name in dir(mod):
        if any(k in name.lower() for k in ("quant", "bitlinear")):
            obj = getattr(mod, name)
            if callable(obj):
                print(f"\n=== {name} ===\n", src(obj))


if __name__ == "__main__":
    main()
