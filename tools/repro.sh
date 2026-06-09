#!/usr/bin/env bash
# ternfpga — one-command reproduction of the simulation suite, model self-tests, and figures.
#
#   bash tools/repro.sh           # RTL sim suite + model encoding test + figures (no model download)
#   bash tools/repro.sh --full    # also re-validate the FFN golden vs PyTorch (needs torch + the model)
#
# Prereqs (a Python venv): cocotb + Verilator (5.020) on PATH, numpy, matplotlib.
#   pip install -r requirements.txt   # plus a system Verilator install
# FPGA synthesis / bitstream / on-board are separate flows — see syn/ and soc/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "================================================================"
echo " ternfpga reproduction — sim suite + model self-tests + figures"
echo "================================================================"

echo
echo "== [1/3] RTL simulation suite (cocotb + Verilator) — every DUT bit-exact vs NumPy golden =="
make -C sim clean
make -C sim all

echo
echo "== [2/3] model encoding self-test (ternary codes + base-3 pack/unpack round-trips) =="
python models/export_weights.py

echo
echo "== [3/3] result figures (energy/token, sparsity, fit sweep, roofline, BRAM fix, gather) =="
python bench/plots/make_plots.py

if [ "${1:-}" = "--full" ]; then
    echo
    echo "== [full] FFN golden vs PyTorch + glue identity (needs torch + microsoft/BitNet-b1.58-2B-4T) =="
    python models/validate_ffn.py --device "${DEVICE:-cpu}"
    python models/ffn_glue_ref.py --device "${DEVICE:-cpu}"
fi

echo
echo "================================================================"
echo " ALL GREEN — ternfpga reproduction complete."
echo "  RTL: every testbench bit-exact (0 DSP).  Figures: bench/plots/*.png"
echo "================================================================"
