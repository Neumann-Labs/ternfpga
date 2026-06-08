# cocotb + verilator simulation suite.
#   make            # run the default test (ternary_dot)
#   make clean
#
# Runs locally (CI) and on worker4. Requires verilator + cocotb on PATH.

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

TOPLEVEL ?= ternary_dot
MODULE   ?= test_ternary_dot

VERILOG_SOURCES += $(shell pwd)/../rtl/ternary_dot.sv

# Allow lint warnings (we keep the RTL clean, but don't fail the sim on style).
EXTRA_ARGS += -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim
