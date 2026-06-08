#!/usr/bin/env bash
# Flash arty_top.bit to the Arty A7-35T over JTAG (volatile SRAM configuration).
#   bash syn/flash.sh [bitstream]
set -euo pipefail
BIT="${1:-/tmp/ternfpga_synth/arty_top.bit}"
[ -f "$BIT" ] || { echo "bitstream not found: $BIT (run syn/build_bitstream.sh first)"; exit 1; }
openFPGALoader -b arty_a7_35t "$BIT"
echo "flashed: $BIT"
