# GPU baseline (RTX 3060)

The GPU side of the energy/token head-to-head: run a model on the 3060 (bf16) and
measure decode tok/s + energy/token (GPU power via `nvidia-smi`).

## Setup (worker4)
NVIDIA driver (DKMS, no kernel change) + a dedicated CUDA-PyTorch venv:
```bash
sudo apt-get install -y nvidia-driver-580 nvidia-utils-580       # DKMS builds for the running kernel
printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo rmmod nouveau && sudo modprobe nvidia nvidia_uvm            # live swap (nouveau refcount 0) — or reboot
nvidia-smi                                                       # confirm the GPU

python3 -m venv /srv/fpga/gpu-venv && . /srv/fpga/gpu-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.46,<5" accelerate                  # NB: transformers 5.x model imports are broken
```
(The driver is pure compute here — an AMD Radeon drives worker4's display — so there
was no display/console risk and no reboot was needed.)

## Run
```bash
python bench/gpu_baseline/run_gpu.py --model microsoft/BitNet-b1.58-2B-4T --n 256   # same model as the CPU baseline
python bench/gpu_baseline/run_gpu.py --model Qwen/Qwen2.5-1.5B-Instruct --n 256     # GPU's best foot (smaller dense)
```

Results + the full head-to-head table → [`../results/gpu_baseline.md`](../results/gpu_baseline.md).
