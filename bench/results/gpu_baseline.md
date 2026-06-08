# GPU baseline (RTX 3060) + the energy/token head-to-head

Measured on worker4's **RTX 3060 12GB** (driver 580.159.03, CUDA 13.0) via
`bench/gpu_baseline/run_gpu.py` (PyTorch bf16 + transformers; decode tok/s and GPU
power sampled from `nvidia-smi` → J/token), 2026-06-08. Batch-1, greedy, 256 tokens.

| platform | model | tok/s | power | **J/token** |
|---|---|---|---|---|
| CPU — Ryzen 9 5950X | BitNet b1.58 2B4T (i2_s **ternary**) | 28.4 | ~121 W (RAPL pkg) | **4.62** |
| **GPU — RTX 3060** | BitNet b1.58 2B4T (**bf16**, dequantized) | 23.5 | 86.4 W | **3.67** |
| GPU — RTX 3060 | Qwen2.5-1.5B-Instruct (bf16) | 57.7 | 105 W | 1.82 |
| FPGA — Arty A7-35T | ternary engine (0 DSP, verified on silicon) | — | **~0.06–0.5 W** | target ~0.25–0.4 |

## The result that matters
On the **same model** (BitNet-2B-4T), the RTX 3060 must **dequantize the ternary
weights to bf16** (4.87 GB resident) — it has no native ternary datapath — and lands
at **3.67 J/token**, only ~1.26× better than the CPU's *native ternary* 4.62 J/token,
and actually **slower** in tok/s (23.5 vs 28.4). The GPU extracts **almost no benefit
from the 1.58-bit weights.** That is precisely the gap a multiply-free, ternary-native
FPGA engine exploits: the dossiers target ~0.25–0.4 J/token on the Arty —
**~10× better than the 3060 on the identical model** — at a sub-watt power envelope
([`power.md`](power.md)) vs the GPU's ~86 W.

The Qwen-1.5B row (1.82 J/token) is the GPU's *best foot forward* on a smaller dense
model; even that is ~5–7× above the FPGA target.

## Honest caveats
- GPU power = `nvidia-smi power.draw` sampled through decode (GPU chip); CPU = RAPL
  package. Both are chip-level dynamic+static during the run — comparable, not identical.
- The FPGA **J/token is still a target**, not an end-to-end measurement — it needs the
  on-board inference datapath (DDR3 streaming, Phase 1). The FPGA **power** (63 mW demo /
  sub-watt envelope) and the **0-DSP multiply-free compute** are measured / silicon-verified.
- Models differ in size (BitNet 2B vs Qwen 1.5B); decode is memory-bandwidth-bound, so
  J/token is a representative envelope. The fair same-model row is the **BitNet GPU number
  (3.67 J/tok)**.
- Single run, batch-1 (the edge/decode regime this project targets); the GPU would pull
  far ahead at large batch — a regime we explicitly do not compete in.

Reproduce: [`../gpu_baseline/README.md`](../gpu_baseline/README.md).
