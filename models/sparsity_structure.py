"""Risk-2 probe: is BitNet's ~60% down_proj activation sparsity genuinely UNSTRUCTURED
(data-dependent per token -> our on-fabric gather is the differentiator), or is the
per-token zero pattern static / N:M-regular enough that a structured router (TENET-style)
or static pruning would capture it (which would collapse the differentiator)?

Hooks every layer's down_proj INPUT, then per layer measures:
  - zero fraction (the sparsity)
  - channel classes: always-zero (static prune), always-active, "sometimes" (DATA-DEPENDENT)
  - token-to-token Jaccard overlap of the active set (1.0 = identical pattern = structured)
  - N:M capture: fraction of true zeros a fixed 50%/block-of-M static mask would predict

  python models/sparsity_structure.py --device cuda
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/BitNet-b1.58-2B-4T"
PROMPT = ("Edge AI inference wants energy efficiency: ternary weights make the multiply a "
          "sign-select, and activation sparsity means most of the work can simply be skipped "
          "per token, which a CPU or GPU cannot exploit but reconfigurable fabric can, so the "
          "question is whether the zeros are predictable or genuinely data dependent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block", type=int, default=32)   # N:M block size
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    load_kw = {} if args.device == "cpu" else {"device_map": args.device}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, **load_kw).eval()

    caps = {}

    def mk(li):
        def h(_m, inp, _out):
            caps[li] = inp[0].detach().float().cpu().numpy()[0]   # (seq, intermediate)
        return h

    for li, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(mk(li))

    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)

    rows = []
    for li in sorted(caps):
        A = caps[li]                       # (T, F)
        T, F = A.shape
        act = A != 0.0
        spars = 1.0 - act.mean()
        colcnt = act.sum(0)                # per-channel active-token count
        always_zero = float((colcnt == 0).mean())
        always_act = float((colcnt == T).mean())
        sometimes = 1.0 - always_zero - always_act      # data-dependent channels
        # token-to-token Jaccard of active sets (exclude self-pairs)
        jac = []
        for i in range(T):
            for j in range(i + 1, T):
                inter = np.logical_and(act[i], act[j]).sum()
                uni = np.logical_or(act[i], act[j]).sum()
                if uni:
                    jac.append(inter / uni)
        jac_mean = float(np.mean(jac)) if jac else 0.0
        # N:M static-mask capture: pick the globally-most-active 50% channels per block; how
        # many of each token's TRUE zeros does that static mask correctly predict-as-zero?
        keep = np.zeros(F, dtype=bool)
        for b in range(0, F, args.block):
            blk = colcnt[b:b + args.block]
            k = max(1, len(blk) // 2)
            idx = np.argsort(blk)[-k:]      # keep the half most-often-active
            keep[b + idx] = True
        pred_zero = ~keep                   # static mask predicts these as zero
        true_zero = ~act
        # fraction of true zeros that the static mask captures (predicted zero AND actually zero)
        capt = (np.logical_and(pred_zero[None, :], true_zero).sum() /
                max(1, true_zero.sum()))
        rows.append((li, spars, always_zero, always_act, sometimes, jac_mean, float(capt)))

    sp = np.array([r[1] for r in rows])
    smt = np.array([r[4] for r in rows])
    jc = np.array([r[5] for r in rows])
    cp = np.array([r[6] for r in rows])
    print(f"model {MODEL}  layers={len(rows)}  tokens={caps[0].shape[0]}  intermediate={caps[0].shape[1]}")
    print("layer  spars  alwaysZ  alwaysA  sometimes  jaccard  N:M-capt")
    for r in rows:
        print(f"{r[0]:5d}  {r[1]*100:5.1f}%  {r[2]*100:6.1f}%  {r[3]*100:6.1f}%  "
              f"{r[4]*100:7.1f}%  {r[5]:7.3f}  {r[6]*100:6.1f}%")
    print(f"\nMEAN  sparsity={sp.mean()*100:.1f}%  sometimes(data-dep)={smt.mean()*100:.1f}%  "
          f"jaccard={jc.mean():.3f}  N:M-capture={cp.mean()*100:.1f}%")
    # Verdict: unstructured if the active set is largely data-dependent (high 'sometimes',
    # low Jaccard) and a static N:M mask captures little of the zeros.
    unstructured = smt.mean() > 0.30 and jc.mean() < 0.85 and cp.mean() < 0.80
    print("VERDICT:", "UNSTRUCTURED (data-dependent — gather differentiator holds)" if unstructured
          else "STRUCTURED-ish (static/N:M could capture much — differentiator weakened)")
    np.save("/tmp/sparsity_rows.npy", np.array(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
