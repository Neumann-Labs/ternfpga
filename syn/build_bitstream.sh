#!/usr/bin/env bash
# Build the Arty A7-35T bitstream for arty_top (synth + place&route + bitstream).
#   bash syn/build_bitstream.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RTL="$(cd "$HERE/../rtl" && pwd)"
XDC="$(cd "$HERE/../constraints" && pwd)/arty_a7_35.xdc"
VSET="${VIVADO_SETTINGS:-/srv/fpga/Xilinx/2025.2/Vivado/settings64.sh}"
OUT="${OUT:-/tmp/ternfpga_synth/arty_top.bit}"

mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC1090
source "$VSET"
cd "$(dirname "$OUT")"
vivado -mode batch -nojournal -log build_bitstream.log \
    -source "$HERE/build_bitstream.tcl" -tclargs "$RTL" "$XDC" "$OUT"
echo "bitstream: $OUT"
