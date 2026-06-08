#!/usr/bin/env bash
# Sync the repo to worker4 (where the toolchain + Arty A7-35T live) and run a
# target there. Authoring happens on the laptop; build/sim/flash on worker4.
#
#   tools/sync_worker4.sh            # default: run the cocotb sim suite
#   tools/sync_worker4.sh sim
#   tools/sync_worker4.sh 'make -C /srv/fpga/ternfpga/sim MODULE=test_foo'
set -euo pipefail

REMOTE="${REMOTE:-worker4}"
DEST="${DEST:-/srv/fpga/ternfpga}"
VENV="${VENV:-/srv/fpga/fpga-spacex/tools/venv}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-sim}"

echo ">> rsync $HERE -> $REMOTE:$DEST"
rsync -az --delete \
  --exclude '.git' --exclude 'sim_build' --exclude '__pycache__' \
  --exclude 'obj_dir' --exclude '*.vcd' --exclude '*.fst' \
  "$HERE"/ "$REMOTE:$DEST/"

case "$TARGET" in
  sim)
    echo ">> remote: make -C $DEST/sim  (verilator + cocotb)"
    ssh "$REMOTE" "source '$VENV/bin/activate' && make -C '$DEST/sim'"
    ;;
  *)
    echo ">> remote: $TARGET"
    ssh "$REMOTE" "source '$VENV/bin/activate' && $TARGET"
    ;;
esac
