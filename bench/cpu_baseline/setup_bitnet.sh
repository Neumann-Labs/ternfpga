#!/usr/bin/env bash
# Reproducible CPU ternary-inference baseline setup (Microsoft bitnet.cpp).
#
# Installs build deps, clones bitnet.cpp, applies two fixes for upstream issues
# hit on Ubuntu 24.04 / clang-18, builds the i2_s ternary kernels + binaries,
# and fetches the official pre-quantized BitNet b1.58 2B4T GGUF. Idempotent.
# Build tree + model live under BITNET_ROOT (default /srv/fpga/bitnet), outside
# this repo (multi-GB, not source).
#
#   bash bench/cpu_baseline/setup_bitnet.sh
#
# Upstream issues worked around (both documented in BUILDLOG):
#   1. ggml-bitnet-mad.cpp has a const-correctness bug clang-18 rejects as an
#      error (older clang/gcc only warn) -> patched in place.
#   2. setup_env.py's HF->GGUF converter doesn't support the 2B-4T arch
#      ('BitNetForCausalLM') -> we ignore that step and pull the official
#      pre-quantized i2_s GGUF instead.
set -euo pipefail
trap 'echo "BITNET_SETUP_EXIT rc=$?"' EXIT

ROOT="${BITNET_ROOT:-/srv/fpga/bitnet}"
VENV="${VENV:-/srv/fpga/fpga-spacex/tools/venv}"
HF_REPO="${HF_REPO:-microsoft/BitNet-b1.58-2B-4T}"
GGUF_REPO="${GGUF_REPO:-microsoft/bitnet-b1.58-2B-4T-gguf}"
QUANT="${QUANT:-i2_s}"

echo "== [1/5] build deps (clang-18, cmake) =="
sudo apt-get update -y >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y clang-18 llvm-18 cmake git >/dev/null
command -v clang   >/dev/null 2>&1 || sudo ln -sf "$(command -v clang-18)"   /usr/local/bin/clang
command -v clang++ >/dev/null 2>&1 || sudo ln -sf "$(command -v clang++-18)" /usr/local/bin/clang++
export CC=clang-18 CXX=clang++-18

echo "== [2/5] clone bitnet.cpp -> $ROOT =="
mkdir -p "$(dirname "$ROOT")"
[ -d "$ROOT/.git" ] || git clone --recursive https://github.com/microsoft/BitNet "$ROOT"

echo "== [3/5] patch upstream const-correctness bug for clang-18 =="
# 'int8_t * y_col = y + ...' where y is 'const int8_t *' -> hard error on clang-18.
sed -i -E 's/^([[:space:]]*)int8_t \* y_col = y/\1const int8_t * y_col = y/' \
  "$ROOT/src/ggml-bitnet-mad.cpp"

echo "== [4/5] python deps + build kernels/binaries =="
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r "$ROOT/requirements.txt"
cd "$ROOT"
# setup_env compiles llama-cli FIRST, then attempts an HF->GGUF convert that
# fails on this model arch. The binaries are built by then, so tolerate the
# nonzero exit and use the pre-quantized GGUF below.
python setup_env.py --hf-repo "$HF_REPO" -q "$QUANT" -md "models/$(basename "$HF_REPO")" \
  || echo "WARN: setup_env nonzero (expected: HF convert unsupported); binaries built, using pre-quantized GGUF"

echo "== [5/5] fetch official pre-quantized i2_s GGUF =="
huggingface-cli download "$GGUF_REPO" --local-dir "models/2B-4T-gguf"

if [ -x build/bin/llama-cli ] && [ -f models/2B-4T-gguf/ggml-model-i2_s.gguf ]; then
  echo "BITNET_SETUP_DONE root=$ROOT gguf=models/2B-4T-gguf/ggml-model-i2_s.gguf"
else
  echo "BITNET_SETUP_INCOMPLETE (missing llama-cli or gguf)"; exit 1
fi
