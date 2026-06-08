# Synthesize + implement arty_top to a bitstream for the Arty A7-35T.
#   vivado -mode batch -source build_bitstream.tcl -tclargs <rtl_dir> <xdc> <out.bit>
set part "xc7a35ticsg324-1L"
set rtl  [lindex $argv 0]
set xdc  [lindex $argv 1]
set out  [lindex $argv 2]

read_verilog -sv "$rtl/ternary_dot.sv"
read_verilog -sv "$rtl/uart_tx.sv"
read_verilog -sv "$rtl/arty_top.sv"
read_xdc $xdc

synth_design -top arty_top -part $part
opt_design
place_design
route_design

report_utilization
report_timing_summary -max_paths 1

write_bitstream -force $out
puts "BITSTREAM_DONE $out"
