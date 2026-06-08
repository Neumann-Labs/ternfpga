# Fit sweep — synthesize ternary_gemv_tiled at a given (K, NT, M) and emit a
# single machine-parsable FITSTAT line plus the full utilization report.
#
# Answers "how big a ternary block fits the Arty A7-35T?": sweeping K (lanes/cycle)
# and KT=K*NT (row width) and M (output rows) shows where register-resident
# activations/outputs blow the LUT/FF budget and force a move to BRAM.
#
#   vivado -mode batch -source fit_sweep.tcl -tclargs <part> <period_ns> <K> <NT> <M> <srcs...>
if {$argc < 6} {
    puts "ERROR: usage: <part> <period_ns> <K> <NT> <M> <srcs...>"
    exit 1
}
set part   [lindex $argv 0]
set period [lindex $argv 1]
set K      [lindex $argv 2]
set NT     [lindex $argv 3]
set M      [lindex $argv 4]
set srcs   [lrange $argv 5 end]

foreach s $srcs { read_verilog -sv $s }
synth_design -top ternary_gemv_tiled -part $part -mode out_of_context \
    -generic K=$K -generic NT=$NT -generic M=$M
create_clock -name clk -period $period [get_ports clk]

set u [report_utilization -return_string]
set luts 0; set ff 0; set dsp 0; set bram 0
regexp {Slice LUTs\*?\s*\|\s*(\d+)}     $u -> luts
regexp {Slice Registers\s*\|\s*(\d+)}   $u -> ff
regexp {DSPs\s*\|\s*(\d+)}              $u -> dsp
regexp {Block RAM Tile\s*\|\s*([0-9.]+)} $u -> bram

# WNS = first signed float appearing after the "WNS(ns)" column header
set t [report_timing_summary -return_string -max_paths 1]
set wns "na"
set idx [string first "WNS(ns)" $t]
if {$idx >= 0} {
    set tail [string range $t $idx end]
    regexp {(-?\d+\.\d+)} $tail -> wns
}

set kt [expr {$K * $NT}]
puts "FITSTAT K=$K NT=$NT M=$M KT=$kt LUT=$luts FF=$ff DSP=$dsp BRAM=$bram WNS=$wns PERIOD=$period"
puts "########## FULL UTILIZATION (K=$K NT=$NT M=$M KT=$kt) ##########"
puts $u
puts "########## DONE ##########"
