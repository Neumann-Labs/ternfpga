#!/usr/bin/env bash
# Reproducible CPU ternary-inference baseline setup (Microsoft bitnet.cpp).
#
# Installs build deps, clones bitnet.cpp, downloads a BitNet b1.58 model, and
# builds its i2_s ternary kernels. Idempotent. Target: an x86-64 Linux box
# (here: worker4). The built tree + model live OUTSIDE the repo (BITNET_ROOT,
# default /srv/fpga/bitnet) since they are multi-GB and not source.
#
#   bash bench/cpu_baseline/setup_bitnet.sh
#   HF_REPO=1bitLLM/bitnet_b1_58-large bash bench/cpu_baseline/setup_bitnet.sh   # smaller 0.7B
set -euo pipefail
trap 'echo "BITNET_SETUP_EXIT rc=$?"' EXIT   # always-fires sentinel for the watcher

ROOT="${BITNET_ROOT:-/srv/fpga/bitnet}"
VENV="${VENV:-/srv/fpga/fpga-spacex/tools/venv}"
HF_REPO="${HF_REPO:-microsoft/BitNet-b1.58-2B-4T}"
QUANT="${QUANT:-i2_s}"

echo "== [1/4] build deps (clang-18, cmake) =="
sudo apt-get update -y >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y clang-18 llvm-18 cmake git >/dev/null
command -v clang   >/dev/null 2>&1 || sudo ln -sf "$(command -v clang-18)"   /usr/local/bin/clang
command -v clang++ >/dev/null 2>&1 || sudo ln -sf "$(command -v clang++-18)" /usr/local/bin/clang++
export CC=clang-18 CXX=clang++-18

echo "== [2/4] clone bitnet.cpp -> $ROOT =="
mkdir -p "$(dirname "$ROOT")"
[ -d "$ROOT/.git" ] || git clone --recursive https://github.com/microsoft/BitNet "$ROOT"

echo "== [3/4] python deps =="
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r "$ROOT/requirements.txt"

echo "== [4/4] download model + build i2_s kernels (long: multi-GB download + clang build) =="
cd "$ROOT"
python setup_env.py --hf-repo "$HF_REPO" -q "$QUANT" -md "models/$(basename "$HF_REPO")"

echo "BITNET_SETUP_DONE repo=$HF_REPO quant=$QUANT root=$ROOT"
