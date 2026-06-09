# Relu-fication upside — the Direction-D ceiling, measured

Direction D skips the zero-activation `down_proj` columns (the gather,
[`down_proj_gather.md`](down_proj_gather.md)); its payoff scales with the FFN activation sparsity.
We measured it on two real models with the same instrument (`models/measure_activation_sparsity.py`,
hooking every `down_proj`):

| model | FFN gate | overall sparsity (measured) | per-layer range | `down_proj` bytes saved (gather) |
|---|---|---:|---|---:|
| **BitNet b1.58 2B-4T** | squared-ReLU | **59.8%** | 42–79% | **56%** |
| **ProSparse-Llama-2-7B** | ReLU (relu-fied) | **83.3%** | 75.6–89.9% | **~80%** |

ProSparse measured with `--model SparseLLM/ProSparse-Llama-2-7B --device cpu --dtype bfloat16
--trust-remote-code` (6 passages, 32 layers, `ffn_dim` 11008; mean **1840 / 11008** active per
token). Our 6-passage mean (83.3%, peak layer 89.9%) is consistent with the ProSparse paper's
**89.3%** headline ([arXiv:2402.13516](https://arxiv.org/abs/2402.13516)). Figure:
[`../plots/sparsity_compare.png`](../plots/sparsity_compare.png).

## What it means
Relu-fication (training the FFN with a ReLU gate — ProSparse / Q-Sparse style) roughly **doubles the
gather payoff**: from ~56% of `down_proj` weight bytes saved at BitNet's 60% sparsity to **~80%** at
ProSparse's 83%. The gather RTL (`sim/tb_gemv_gather.py`) already exploits whatever sparsity the
model provides — **bit-exact, engine unchanged** — so Direction D is **~2.5× on stock BitNet → ~5×
with relu-fication**, on *per-token, unstructured* sparsity a GPU's dense / 2:4 array can't touch.

A **relu-fied BitNet** (ternary weights *and* ~85% activation sparsity — future work, a fine-tune)
would combine 1.6-bit weights with the ProSparse fetch reduction: the full **Direction A × D** stack.
This measurement quantifies that ceiling on real models; the FPGA datapath to capture it is already
built (the column-sparse gather).
