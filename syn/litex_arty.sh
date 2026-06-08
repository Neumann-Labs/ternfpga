#!/usr/bin/env bash
# Build + flash the LiteX Arty A7-35T SoC (VexRiscv + LiteDRAM DDR3 + BIOS).
# Proves DDR3 calibration on the board via the BIOS memtest over UART — the
# Phase-1 memory foundation. See bench/results/ddr3_onboard.md.
set -euo pipefail

ROOT="${LITEX_ROOT:-/srv/fpga/litex}"            # litex_setup.py clones the ecosystem here
BUILD="${LITEX_BUILD:-/srv/fpga/litex-build}"    # build cwd — NOT $ROOT (it shadows the 'litex' pkg)
VENV="${VENV:-/srv/fpga/fpga-spacex/tools/venv}"
VSET="${VIVADO_SETTINGS:-/srv/fpga/Xilinx/2025.2/Vivado/settings64.sh}"
BIT="$BUILD/build/digilent_arty/gateware/digilent_arty.bit"

# ---- one-time setup (uncomment to run) -------------------------------------
# sudo apt-get install -y gcc-riscv64-unknown-elf                 # BIOS toolchain
# source "$VENV/bin/activate"
# mkdir -p "$ROOT" && cd "$ROOT" \
#   && wget -q https://raw.githubusercontent.com/enjoy-digital/litex/master/litex_setup.py \
#   && python litex_setup.py --init --install                     # litex + cores + litex-boards
# pip install meson ninja                                         # LiteX BIOS build system
# ----------------------------------------------------------------------------

# shellcheck disable=SC1090
source "$VENV/bin/activate"
# shellcheck disable=SC1090
source "$VSET"
mkdir -p "$BUILD"; cd "$BUILD"                   # run from a neutral dir (avoids shadowing 'litex')
python -m litex_boards.targets.digilent_arty --variant a7-35 --build

echo "bitstream: $BIT"
echo "flash:  openFPGALoader -b arty_a7_35t $BIT"
echo "uart :  stty -F /dev/ttyUSB1 115200 raw -echo; cat /dev/ttyUSB1    # expect 'Memtest OK'"
