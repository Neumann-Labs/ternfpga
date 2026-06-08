#!/usr/bin/env python3
"""GPU baseline: run a model on the RTX 3060, measure decode tok/s + energy/token.

The GPU side of the energy/token head-to-head. Loads a model (default the same
BitNet b1.58 2B4T used for the CPU baseline; the GPU runs it dequantized to
bf16 — GPUs can't do ternary natively, which is the whole thesis), generates N
tokens greedily, and samples GPU power (nvidia-smi) throughout to get J/token.

  python bench/gpu_baseline/run_gpu.py --model microsoft/BitNet-b1.58-2B-4T --n 256
  python bench/gpu_baseline/run_gpu.py --model Qwen/Qwen2.5-1.5B-Instruct --n 256   # fallback 2B-class
"""
import argparse
import subprocess
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _poll_power(stop_evt, samples, period=0.05):
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                text=True,
            )
            samples.append(float(out.strip().splitlines()[0]))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/BitNet-b1.58-2B-4T")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompt", default="The future of efficient on-device AI inference is")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    ).eval()

    ids = tok(args.prompt, return_tensors="pt").to("cuda")
    n_in = ids["input_ids"].shape[1]

    with torch.no_grad():                                  # warm-up
        model.generate(**ids, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    samples, stop = [], threading.Event()
    poller = threading.Thread(target=_poll_power, args=(stop, samples))
    poller.start()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.n, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop.set()
    poller.join()

    gen = out.shape[1] - n_in
    avg_p = sum(samples) / len(samples) if samples else float("nan")
    tps = gen / dt
    jpt = avg_p * dt / gen if gen else float("nan")
    mem = torch.cuda.max_memory_allocated() / 1e9

    print("===== GPU BASELINE (RTX 3060) =====")
    print(f"model        : {args.model}")
    print(f"dtype        : bfloat16   vram_peak_GB: {mem:.2f}")
    print(f"gen_tokens   : {gen}    wall_s: {dt:.2f}")
    print(f"decode_tok/s : {tps:.1f}")
    print(f"avg_power_W  : {avg_p:.1f}   (nvidia-smi power.draw, {len(samples)} samples)")
    print(f"J/token      : {jpt:.3f}")
    print("===================================")


if __name__ == "__main__":
    main()
