# Simulation suite — cocotb + Verilator, one target per DUT.
#
#   make            # build + run every testbench
#   make dot|gemv   # a single DUT
#   make clean
#
# We drive cocotb's classic Makefile.sim once per testbench (isolated build
# dirs + result files). NOTE: cocotb 1.9's experimental `cocotb.runner` could
# not bind a top module that instantiates a submodule on Verilator 5.020, so we
# use the rock-solid Makefile flow. (Revisit with Verilator 5.036+/cocotb 2.0 —
# see the toolchain-modernization issue.)
#
# `.RECIPEPREFIX := >` lets recipes use '>' instead of a leading TAB.

.RECIPEPREFIX := >

TOPDIR := $(shell pwd)
RTL    := $(TOPDIR)/../rtl
MKSIM  := $(shell cocotb-config --makefiles)/Makefile.sim
COMMON := SIM=verilator TOPLEVEL_LANG=verilog EXTRA_ARGS=-Wno-fatal

# tb_*.py (here) and the NumPy golden (../models) must be importable by the sim.
export PYTHONPATH := $(TOPDIR):$(TOPDIR)/../models

.PHONY: all dot gemv clean

all: dot gemv

dot:
> @echo "=== ternary_dot ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_dot MODULE=tb_ternary_dot VERILOG_SOURCES="$(RTL)/ternary_dot.sv" SIM_BUILD=sim_build/ternary_dot COCOTB_RESULTS_FILE=results_dot.xml

gemv:
> @echo "=== ternary_gemv ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv MODULE=tb_ternary_gemv VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv.sv" SIM_BUILD=sim_build/ternary_gemv COCOTB_RESULTS_FILE=results_gemv.xml

clean:
> rm -rf sim_build results_*.xml __pycache__ .pytest_cache
