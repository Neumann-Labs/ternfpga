#!/usr/bin/env bash
# Post-route vectorless power estimate for arty_top on the Arty A7-35T.
#   bash syn/report_power.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RTL="$(cd "$HERE/../rtl" && pwd)"
XDC="$(cd "$HERE/../constraints" && pwd)/arty_a7_35.xdc"
VSET="${VIVADO_SETTINGS:-/srv/fpga/Xilinx/2025.2/Vivado/settings64.sh}"
OUT="${OUT:-/tmp/ternfpga_synth}"
# shellcheck disable=SC1090
source "$VSET"
mkdir -p "$OUT"; cd "$OUT"
vivado -mode batch -nojournal -log report_power.log -source "$HERE/report_power.tcl" -tclargs "$RTL" "$XDC"
echo "=== power ($OUT/power_arty.rpt) ==="
grep -E "Total On-Chip Power|Dynamic \(|Device Static|Confidence" "$OUT/power_arty.rpt" | head
