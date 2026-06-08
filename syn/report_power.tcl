# Vectorless post-route power estimate for arty_top on the Arty A7-35T.
#   vivado -mode batch -source report_power.tcl -tclargs <rtl_dir> <xdc>
set part "xc7a35ticsg324-1L"
set rtl  [lindex $argv 0]
set xdc  [lindex $argv 1]

read_verilog -sv "$rtl/ternary_dot.sv"
read_verilog -sv "$rtl/uart_tx.sv"
read_verilog -sv "$rtl/arty_top.sv"
read_xdc $xdc

synth_design -top arty_top -part $part
opt_design
place_design
route_design

report_power -file power_arty.rpt
puts "POWER_REPORT_DONE"
