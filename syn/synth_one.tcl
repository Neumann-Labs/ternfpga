# Out-of-context Vivado synthesis of one ternfpga module.
# Emits utilization (LUT/FF/DSP/BRAM/CARRY) and, for clocked modules, a timing
# summary against a target clock so we can read Fmax margin (WNS).
#
#   vivado -mode batch -source synth_one.tcl -tclargs <part> <top> <period_ns> <src.sv>...
if {$argc < 4} {
    puts "ERROR: usage: <part> <top> <period_ns> <srcs...>"
    exit 1
}
set part   [lindex $argv 0]
set top    [lindex $argv 1]
set period [lindex $argv 2]
set srcs   [lrange $argv 3 end]

foreach s $srcs { read_verilog -sv $s }
synth_design -top $top -part $part -mode out_of_context

set has_clk [expr {[llength [get_ports -quiet clk]] > 0}]
if {$has_clk} {
    create_clock -name clk -period $period [get_ports clk]
}

puts "########## UTILIZATION $top ##########"
report_utilization
if {$has_clk} {
    puts "########## TIMING $top (target period ${period} ns) ##########"
    report_timing_summary -max_paths 1
}
puts "########## DONE $top ##########"
