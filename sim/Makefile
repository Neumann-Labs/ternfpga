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

.PHONY: all dot gemv sparse pipe unpack packed pe gemvpipe feed tile tiled stream ffn gather bench uart arty clean

all: dot gemv sparse pipe unpack packed pe gemvpipe feed tile tiled stream ffn gather bench uart arty

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

pe:
> @echo "=== ternary_pe_array ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_pe_array MODULE=tb_ternary_pe_array VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_pe_array.sv" SIM_BUILD=sim_build/ternary_pe_array COCOTB_RESULTS_FILE=results_pe.xml

gemvpipe:
> @echo "=== ternary_gemv_pipe ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_pipe MODULE=tb_ternary_gemv_pipe VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_pipe.sv" SIM_BUILD=sim_build/ternary_gemv_pipe COCOTB_RESULTS_FILE=results_gemvpipe.xml

feed:
> @echo "=== weight_feed ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=weight_feed MODULE=tb_weight_feed VERILOG_SOURCES="$(RTL)/ternary_unpack5.sv $(RTL)/weight_feed.sv" SIM_BUILD=sim_build/weight_feed COCOTB_RESULTS_FILE=results_feed.xml

tile:
> @echo "=== ternary_tile (mem->engine) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_tile MODULE=tb_ternary_tile VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_pipe.sv $(RTL)/ternary_unpack5.sv $(RTL)/weight_feed.sv $(RTL)/ternary_tile.sv" SIM_BUILD=sim_build/ternary_tile COCOTB_RESULTS_FILE=results_tile.xml

tiled:
> @echo "=== ternary_gemv_tiled ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_tiled MODULE=tb_ternary_gemv_tiled VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv_tiled.sv" SIM_BUILD=sim_build/ternary_gemv_tiled COCOTB_RESULTS_FILE=results_tiled.xml

stream:
> @echo "=== ternary_gemv_stream (BRAM-centric, pipelined) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_stream MODULE=tb_ternary_gemv_stream VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_stream.sv" SIM_BUILD=sim_build/ternary_gemv_stream COCOTB_RESULTS_FILE=results_stream.xml

ffn:
> @echo "=== ffn_block (full FFN end-to-end via the stream GEMV) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_stream MODULE=tb_ffn_block VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_stream.sv" SIM_BUILD=sim_build/ffn_block COCOTB_RESULTS_FILE=results_ffn.xml

gather:
> @echo "=== gemv_gather (down_proj column-sparse activation gather) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_stream MODULE=tb_gemv_gather VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_stream.sv" SIM_BUILD=sim_build/gemv_gather COCOTB_RESULTS_FILE=results_gather.xml

bench:
> @echo "=== ternary_gemv_bench (on-silicon throughput harness) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv_bench MODULE=tb_ternary_gemv_bench VERILOG_SOURCES="$(RTL)/ternary_dot_pipe.sv $(RTL)/ternary_gemv_stream.sv $(RTL)/ternary_gemv_bench.sv" SIM_BUILD=sim_build/ternary_gemv_bench COCOTB_RESULTS_FILE=results_bench.xml

uart:
> @echo "=== uart_tx ==="
# NOTE: pass the -G parameter override via EXTRA_ARGS, not COMPILE_ARGS — the
# latter would clobber cocotb's internal `--prefix Vtop` and break the build.
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=uart_tx MODULE=tb_uart_tx VERILOG_SOURCES="$(RTL)/uart_tx.sv" EXTRA_ARGS="-Wno-fatal -GCLKS_PER_BIT=8" SIM_BUILD=sim_build/uart_tx COCOTB_RESULTS_FILE=results_uart.xml

arty:
> @echo "=== arty_top (integration) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=arty_top MODULE=tb_arty_top VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/uart_tx.sv $(RTL)/arty_top.sv" EXTRA_ARGS="-Wno-fatal -GCLKS_PER_BIT=8" SIM_BUILD=sim_build/arty_top COCOTB_RESULTS_FILE=results_arty.xml

# Real BitNet ternary weights (needs models/data/real_tile.npz; committed fixture).
real:
> @echo "=== ternary_gemv (real BitNet tile) ==="
> @$(MAKE) -f $(MKSIM) $(COMMON) TOPLEVEL=ternary_gemv MODULE=tb_gemv_from_file VERILOG_SOURCES="$(RTL)/ternary_dot.sv $(RTL)/ternary_gemv.sv" SIM_BUILD=sim_build/ternary_gemv_real COCOTB_RESULTS_FILE=results_real.xml

clean:
> rm -rf sim_build results_*.xml __pycache__ .pytest_cache
