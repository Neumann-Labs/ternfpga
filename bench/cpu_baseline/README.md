# CPU ternary-inference baseline (bitnet.cpp)

The first *measured* reference point for the whole project: how fast and how
energy-hungry is ternary (BitNet b1.58) LLM decode on a strong CPU? This anchors
the FPGA's eventual energy/token claim and de-risks the model/tooling **before**
any GPU (it needs none) and before the FPGA exists.

## What it does
- `setup_bitnet.sh` — installs deps (clang-18, cmake), clones Microsoft's
  [bitnet.cpp](https://github.com/microsoft/BitNet), downloads a BitNet b1.58
  model, and builds the `i2_s` ternary kernels. Build tree + model live under
  `$BITNET_ROOT` (default `/srv/fpga/bitnet`), outside this repo (multi-GB).
- `run_baseline.sh` — runs decode and reports **tokens/sec** (llama eval timing)
  and **energy/token** (CPU-package RAPL via `/sys/class/powercap/intel-rapl:0`).

## Run (on worker4, or any x86-64 Linux with the venv)
```bash
bash bench/cpu_baseline/setup_bitnet.sh                 # ~minutes: download + clang build
bash bench/cpu_baseline/run_baseline.sh                 # prints the baseline numbers
# smaller/faster model (0.7B, closer to our on-Arty scale):
HF_REPO=1bitLLM/bitnet_b1_58-large bash bench/cpu_baseline/setup_bitnet.sh
```

Knobs (env): `BITNET_ROOT`, `HF_REPO`, `QUANT` (`i2_s`/`tl2`), `N_TOKENS`,
`THREADS`, `MODEL`, `PROMPT`.

## Caveats (honesty matters here)
- RAPL energy is **whole CPU package**, integrated over the **full run** (prefill
  + decode) ÷ generated tokens — an honest *upper bound* on J/token. A
  decode-only figure (subtracting prefill + idle) comes in a later refinement.
- This is the **CPU** baseline. The GPU (RTX 3060) baseline needs the NVIDIA
  driver + CUDA (a coordinated worker4 maintenance step — see the build plan);
  it is intentionally deferred. The CPU number is useful on its own and is what
  bitnet.cpp itself reports speedups against.

Results are recorded in [`../results/cpu_baseline.md`](../results/cpu_baseline.md).
