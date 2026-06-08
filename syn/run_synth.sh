#!/usr/bin/env bash
# Out-of-context Vivado synthesis of the ternfpga RTL on the Arty A7-35T part.
# Prints LUT/FF/DSP/BRAM utilization + Fmax-relevant timing (WNS) per module.
# Full Vivado logs land in $OUT (default /tmp/ternfpga_synth).
#
#   bash syn/run_synth.sh
#   PART=xc7a35tcsg324-1 PERIOD=5.0 bash syn/run_synth.sh
set -euo pipefail

PART="${PART:-xc7a35ticsg324-1L}"   # Digilent Arty A7-35T part
PERIOD="${PERIOD:-4.0}"             # 250 MHz target; WNS shows margin -> Fmax
HERE="$(cd "$(dirname "$0")" && pwd)"
RTL="$(cd "$HERE/../rtl" && pwd)"
VSET="${VIVADO_SETTINGS:-/srv/fpga/Xilinx/2025.2/Vivado/settings64.sh}"
OUT="${OUT:-/tmp/ternfpga_synth}"
TCL="$HERE/synth_one.tcl"

# shellcheck disable=SC1090
source "$VSET"
mkdir -p "$OUT"; cd "$OUT"

run() {
    local top="$1"; shift
    echo "===== synth $top  (part $PART, target ${PERIOD} ns) ====="
    if ! vivado -mode batch -nojournal -log "synth_${top}.log" -source "$TCL" \
            -tclargs "$PART" "$top" "$PERIOD" "$@" >/dev/null 2>&1; then
        echo "  vivado FAILED for $top (tail of $OUT/synth_${top}.log):"
        tail -15 "$OUT/synth_${top}.log"
        return 1
    fi
    grep -E "Slice LUTs|CLB LUTs|Slice Registers|CLB Registers|\| DSPs|Block RAM Tile|CARRY|Bonded IOB" \
        "synth_${top}.log" | sed 's/^/    /' | head -10
    grep -m1 -E "WNS\(ns\)|Design Timing Summary" -A2 "synth_${top}.log" 2>/dev/null | sed 's/^/    /' || true
    echo
}

run ternary_dot          "$RTL/ternary_dot.sv"
run ternary_gemv         "$RTL/ternary_dot.sv" "$RTL/ternary_gemv.sv"
run ternary_gemv_sparse  "$RTL/ternary_dot.sv" "$RTL/ternary_gemv_sparse.sv"
run ternary_dot_pipe     "$RTL/ternary_dot_pipe.sv"
run ternary_unpack5      "$RTL/ternary_unpack5.sv"
run ternary_gemv_packed  "$RTL/ternary_dot.sv" "$RTL/ternary_unpack5.sv" "$RTL/ternary_gemv_packed.sv"
run ternary_pe_array     "$RTL/ternary_dot.sv" "$RTL/ternary_pe_array.sv"
run ternary_gemv_pipe    "$RTL/ternary_dot_pipe.sv" "$RTL/ternary_gemv_pipe.sv"
run weight_feed          "$RTL/ternary_unpack5.sv" "$RTL/weight_feed.sv"

echo "DONE — full Vivado logs in $OUT/"
