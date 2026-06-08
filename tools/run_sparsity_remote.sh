#!/usr/bin/env bash
# Remote launcher (worker4): ensure deps, then measure BitNet FFN activation sparsity.
# Run detached under tmux so a flaky SSH link can't interrupt it:
#   tmux new-session -d -s ternsparsity "bash tools/run_sparsity_remote.sh > /tmp/sparsity.log 2>&1"
set -e
VENV=/srv/fpga/fpga-spacex/tools/venv
cd /srv/fpga/ternfpga
echo "[run_sparsity] ensuring accelerate ..."
"$VENV/bin/pip" install -q accelerate
echo "[run_sparsity] measuring ..."
"$VENV/bin/python" models/measure_activation_sparsity.py
echo "[run_sparsity] done."
