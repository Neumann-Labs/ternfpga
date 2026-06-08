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

.PHONY: all dot gemv sparse pipe unpack packed uart clean

all: dot gemv sparse pipe unpack packed uart

dot:
> @echo "=== ternary_dot ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_dot MODULE=tb_ternary_dot VERILOG_SOURCES="$(RTL)/ternary_dot.sv" SIM_BUILD=sim_build/ternary_dot COCOTB_RESULTS_FILE=results_dot.xml

gemv:
> @echo "=== ternary_gemv ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv MODULE=tb_ternary_gemv VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv.sv" SIM_BUILD=sim_build/ternary_gemv COCOTB_RESULTS_FILE=results_gemv.xml

sparse:
> @echo "=== ternary_gemv_sparse ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_sparse MODULE=tb_ternary_gemv_sparse VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv_sparse.sv" SIM_BUILD=sim_build/ternary_gemv_sparse COCOTB_RESULTS_FILE=results_sparse.xml

pipe:
> @echo "=== ternary_dot_pipe ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_dot_pipe MODULE=tb_ternary_dot_pipe VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv" SIM_BUILD=sim_build/ternary_dot_pipe COCOTB_RESULTS_FILE=results_pipe.xml

unpack:
> @echo "=== ternary_unpack5 ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_unpack5 MODULE=tb_ternary_unpack5 VERILOG_SOURCES="$(RTL)/ternary_unpack5.sv" SIM_BUILD=sim_build/ternary_unpack5 COCOTB_RESULTS_FILE=results_unpack.xml

packed:
> @echo "=== ternary_gemv_packed ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_packed MODULE=tb_ternary_gemv_packed VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_unpack5.sv $(RTL)/ternary_gemv_packed.sv" SIM_BUILD=sim_build/ternary_gemv_packed COCOTB_RESULTS_FILE=results_packed.xml

uart:
> @echo "=== uart_tx ==="
# NOTE: pass the -G parameter override via EXTRA_ARGS, not COMPILE_ARGS — the
# latter would clobber cocotb's internal `--prefix Vtop` and break the build.
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=uart_tx MODULE=tb_uart_tx VERILOG_SOURCES="$(RTL)/uart_tx.sv" EXTRA_ARGS="-Wno-fatal -GCLKS_PER_BIT=8" SIM_BUILD=sim_build/uart_tx COCOTB_RESULTS_FILE=results_uart.xml

# Real BitNet ternary weights (needs models/data/real_tile.npz; committed fixture).
real:
> @echo "=== ternary_gemv (real BitNet tile) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv MODULE=tb_gemv_from_file VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv.sv" SIM_BUILD=sim_build/ternary_gemv_real COCOTB_RESULTS_FILE=results_real.xml

clean:
> rm -rf sim_build results_*.xml __pycache__ .pytest_cache
