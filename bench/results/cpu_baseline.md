# CPU baseline — BitNet b1.58 2B4T (i2_s) on Ryzen 9 5950X

Measured with bitnet.cpp (`bench/cpu_baseline/`) on worker4, 2026-06-08.
Reproduce: `bash bench/cpu_baseline/setup_bitnet.sh && bash bench/cpu_baseline/run_baseline.sh`.

| metric | value |
|---|---|
| model | BitNet b1.58 2B4T, i2_s ternary (1.19 GB GGUF) |
| host | AMD Ryzen 9 5950X (16C/32T), 16 threads |
| decode throughput | **28.4 tokens/s** |
| wall time (256 tok, incl. prefill) | 9.73 s |
| CPU package energy (RAPL, full run) | 1182.6 J |
| avg package power | ~121 W |
| **energy per token** | **~4.62 J/token** |

**Caveats (honesty).** Energy is whole-CPU-package via RAPL (`intel-rapl:0`),
integrated over the full run (prefill + 256-token decode) ÷ 256 — an honest
*upper bound* on decode J/token. A decode-only figure (subtracting prefill +
idle draw) will be lower; tracked for a later refinement. Greedy decode, fixed
prompt, single run.

**How it frames the FPGA target.** This is the strong-CPU reference point. The
energy thesis (`docs/A-ternary-engine.md`, `docs/D-sparsity.md`) targets the
Arty A7-35T at ~0.25–0.4 J/token on a ~300M ternary model — roughly **10× better
energy/token than this 16-core desktop CPU** on a 2B model. The eventual
head-to-head is vs the RTX 3060 (projected ~1–2 J/token); that GPU baseline is
the one step needing the NVIDIA driver, and is deferred. This CPU number anchors
the low end of "what efficient ternary inference costs on commodity silicon."

**Build notes.** Two upstream issues were worked around (both automated in
`setup_bitnet.sh`): a clang-18 const-correctness error in `ggml-bitnet-mad.cpp`
(patched), and an unsupported HF→GGUF conversion for the 2B-4T arch
(`BitNetForCausalLM`) — used the official pre-quantized GGUF instead. See
BUILDLOG cycle 4.
