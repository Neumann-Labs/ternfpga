"""Measure per-token activation sparsity in BitNet b1.58's FFN — the Direction-D lever.

Direction D hinges on per-token *unstructured activation* sparsity: BitNet b1.58's
FFN is a squared-ReLU gate, `down_proj( ReLU(gate_proj(x))^2 . up_proj(x) )`, so the
vector feeding each `down_proj` is exactly zero wherever `gate_proj(x) <= 0`. Every
zero there is a weight *column* of `down_proj` that never has to be fetched or
multiplied — a byte-traffic and FLOP saving a GPU can't exploit (it only does rigid
2:4 structured sparsity).

Our deep-research sweep (docs/research/scaling-feasibility.md, angle 4) found NO
published figure for BitNet b1.58 specifically — so we measure it ourselves. We hook
every `down_proj`, push diverse real text through one forward each, and report the
zero fraction overall and per layer, plus the per-token active-count distribution
(which sizes the hardware gather buffer / index queue).

  python models/measure_activation_sparsity.py --model microsoft/BitNet-b1.58-2B-4T

Measures *activation* sparsity (this file). For *weight* (ternary) sparsity see
extract_bitnet_layer.py. Both feed the streaming-traffic model; activation sparsity
is the unverified, higher-upside one.
"""
from __future__ import annotations

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Diverse passages — prose, code, math, dialogue, factual — so the measured sparsity
# reflects real mixed-domain decode, not one narrow distribution.
PASSAGES = [
    "The future of efficient on-device AI inference is being shaped by extreme "
    "quantization. When weights are constrained to just three values, minus one, "
    "zero, and plus one, the multiply collapses into a simple select-and-add, and "
    "the energy cost of moving bytes from memory dominates everything else.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = "
    "arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = "
    "[x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n",
    "Consider a matrix-vector product y = W x where W has shape M by K. For batch-one "
    "decode the arithmetic intensity is roughly one floating point operation per byte, "
    "so the whole weight matrix is streamed from memory once per token and bandwidth, "
    "not compute, sets the ceiling on tokens per second.",
    "\"Why does the FPGA win on energy per token if it loses on raw throughput?\" she "
    "asked. \"Because,\" he replied, \"a GPU has to dequantize the ternary weights back "
    "to sixteen-bit floats before its tensor cores can touch them, so it pays for "
    "precision it then throws away. The FPGA never leaves the ternary domain.\"",
    "In 1958 the perceptron was introduced as a model of biological learning. Decades "
    "later, deep neural networks with billions of parameters would dominate language "
    "modeling, yet the core operation remained the same: a weighted sum of inputs "
    "followed by a nonlinearity. The choice of that nonlinearity governs how sparse "
    "the intermediate activations become.",
    "The Arty A7-35T is a low-cost FPGA development board built around a Xilinx Artix-7 "
    "device with roughly twenty thousand lookup tables, ninety DSP slices, two hundred "
    "twenty five kilobytes of block RAM, and a single sixteen-bit DDR3 memory chip. It "
    "costs about one hundred thirty dollars and draws under a watt for small designs.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/BitNet-b1.58-2B-4T")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (sparsity is dtype-invariant)")
    ap.add_argument("--dtype", default="auto", help="auto|float32|float16|bfloat16 (use float16 for big CPU loads)")
    ap.add_argument("--eps", type=float, default=1e-8, help="|x|<eps counts as zero")
    ap.add_argument("--out", default="bench/results/activation_sparsity.md")
    args = ap.parse_args()

    if args.dtype != "auto":
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    else:
        dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    load_kw = {} if args.device == "cpu" else {"device_map": args.device}
    print(f"loading {args.model} (dtype {dtype}, device {args.device}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=args.trust_remote_code, **load_kw,
    ).eval()

    # FFN second matmul: its input is the squared-ReLU-gated activation we care about.
    targets = {n: m for n, m in model.named_modules() if n.endswith("down_proj")}
    assert targets, "no '*.down_proj' modules found — check the model's MLP naming"
    print(f"hooking {len(targets)} down_proj layers: e.g. {next(iter(targets))}", flush=True)

    agg = {n: {"zeros": 0, "total": 0, "active_frac": []} for n in targets}

    def mk_hook(name):
        def hook(_mod, inp):
            x = inp[0].detach()                       # [batch, seq, ffn_dim]
            z = x.abs() < args.eps
            agg[name]["zeros"] += int(z.sum())
            agg[name]["total"] += int(x.numel())
            agg[name]["active_frac"].extend((~z).float().mean(dim=-1).flatten().tolist())
        return hook

    handles = [m.register_forward_pre_hook(mk_hook(n)) for n, m in targets.items()]
    with torch.no_grad():
        for i, p in enumerate(PASSAGES):
            ids = tok(p, return_tensors="pt").to(model.device)
            model(**ids)
            print(f"  passage {i+1}/{len(PASSAGES)}: {ids['input_ids'].shape[1]} tokens", flush=True)
    for h in handles:
        h.remove()

    # ---- aggregate ----------------------------------------------------------
    def layer_idx(name):                              # ...layers.<i>.mlp.down_proj
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return -1

    rows = []
    tot_z = tot_n = 0
    all_active = []
    for name, d in sorted(agg.items(), key=lambda kv: layer_idx(kv[0])):
        spars = d["zeros"] / d["total"] if d["total"] else float("nan")
        rows.append((layer_idx(name), spars, d["active_frac"]))
        tot_z += d["zeros"]
        tot_n += d["total"]
        all_active.extend(d["active_frac"])

    overall = tot_z / tot_n if tot_n else float("nan")
    ffn_dim = model.config.intermediate_size
    layer_spars = [r[1] for r in rows]

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    # active *count* per token (mean / p95 / max) sizes the gather buffer
    active_counts = [f * ffn_dim for f in all_active]
    mean_active = statistics.mean(active_counts) if active_counts else float("nan")
    p95_active = pct(active_counts, 0.95)
    max_active = max(active_counts) if active_counts else float("nan")

    print("\n===== BitNet b1.58 FFN ACTIVATION SPARSITY =====")
    print(f"model            : {args.model}")
    print(f"ffn_dim          : {ffn_dim}   layers: {len(rows)}   tokens*layers: {len(all_active)}")
    print(f"OVERALL sparsity : {overall*100:.1f}%  (fraction of down_proj inputs == 0)")
    print(f"per-layer sparsity: min {min(layer_spars)*100:.1f}%  "
          f"mean {statistics.mean(layer_spars)*100:.1f}%  max {max(layer_spars)*100:.1f}%")
    print(f"active units/token: mean {mean_active:.0f}  p95 {p95_active:.0f}  max {max_active:.0f}  "
          f"of {ffn_dim}")
    print("================================================\n")

    # ---- write report -------------------------------------------------------
    lines = [
        "# BitNet b1.58 FFN Activation Sparsity (measured)",
        "",
        f"**Model:** `{args.model}` · {dtype} {args.device} forward · {len(PASSAGES)} diverse passages · "
        f"`|x| < {args.eps}` counts as zero.",
        "",
        "Measures the per-token zero fraction of the vector feeding each FFN `down_proj` — "
        "i.e. the squared-ReLU-gated activation `ReLU(gate_proj(x))^2 . up_proj(x)`. Every zero "
        "is a `down_proj` weight column that need not be fetched or multiplied (Direction D). "
        "This is the figure our deep-research sweep could not find published for b1.58 "
        "specifically (`docs/research/scaling-feasibility.md`, angle 4).",
        "",
        "## Headline",
        "",
        f"- **Overall activation sparsity: {overall*100:.1f}%**  "
        f"(mean per-layer {statistics.mean(layer_spars)*100:.1f}%, "
        f"range {min(layer_spars)*100:.1f}%–{max(layer_spars)*100:.1f}%).",
        f"- **Active units per token:** mean **{mean_active:.0f}**, p95 **{p95_active:.0f}**, "
        f"max **{max_active:.0f}** of **{ffn_dim}** → sizes the hardware gather buffer / index queue.",
        f"- Sample size: {len(all_active):,} (token × layer) activation vectors, "
        f"{tot_n:,} scalar activations.",
        "",
        "## Per-layer",
        "",
        "| layer | activation sparsity | mean active units / token |",
        "|------:|--------------------:|--------------------------:|",
    ]
    for idx, spars, af in rows:
        mean_a = statistics.mean(af) * ffn_dim if af else float("nan")
        lines.append(f"| {idx} | {spars*100:.1f}% | {mean_a:.0f} |")
    lines += [
        "",
        "## What it means for the build",
        "",
        f"- **Measured {overall*100:.0f}% sparsity ({(1-overall)*100:.0f}% active) — NOT the 85–95% "
        f"Direction D assumed.** BitNet b1.58's squared-ReLU gate zeros ~{overall*100:.0f}% of "
        f"`down_proj` inputs per token, so an activation-gather path fetches/multiplies only "
        f"~{(1-overall)*100:.0f}% of `down_proj`'s weight columns — a real ~{1/(1-overall):.1f}× cut "
        "in `down_proj` traffic and FLOPs that a GPU cannot exploit (it does only rigid 2:4 "
        "structured sparsity). But it is **not** the 10–20× the README's Direction D claimed; that "
        "figure came from separate relu-fication / ProSparse work, not stock b1.58.",
        "- **Sparsity is uneven by depth** — early layers (1–8) are ~65–79% sparse (skip most), "
        "middle layers (~11–16) only ~42–47% (little to skip). A static gather sized for the worst "
        f"case (p95 active ≈ {p95_active:.0f} of {ffn_dim}) wastes little.",
        "- **Path to higher sparsity:** relu-fication / ProSparse fine-tuning reaches 85–95% FFN "
        "sparsity in the literature — a future lever if Direction D's payoff must grow. Until then, "
        f"claim only the measured ~{overall*100:.0f}%.",
        f"- **Gather cost:** the index queue need only hold ~p95 active entries (~{p95_active:.0f}), "
        f"a modest BRAM cost; the open question is whether gather control is cheaper than streaming "
        f"all {ffn_dim} columns dense at {(1-overall)*100:.0f}% density.",
        "",
        "_Reproduce:_ `python models/measure_activation_sparsity.py --device cuda`",
    ]
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
