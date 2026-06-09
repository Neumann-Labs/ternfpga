"""LiteX CSR peripheral wrapping ternary_gemv_bench — the on-silicon throughput harness.

Loads the activation (x) and the weight tiles (w_mem) into the engine's resident BRAMs via
CSRs, pulses `run`, and the replay FSM streams the weights at 1 tile/cycle; the hardware
`cycle_count` captures the exact compute latency (run -> done) — the measured number. K=8
keeps `x_wdata` a clean 64-bit CSR.
"""
import math

from migen import Module, Signal, ClockSignal, ResetSignal, Instance
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

RTL_FILES = ["ternary_dot_pipe.sv", "ternary_gemv_stream.sv", "ternary_gemv_bench.sv"]


class TernaryGemvBench(Module, AutoCSR):
    def __init__(self, platform, rtl_dir, K=8, NT_MAX=64, M_MAX=64, WDEPTH=4096):
        aw = max(1, math.ceil(math.log2(NT_MAX)))
        mw = max(1, math.ceil(math.log2(M_MAX)))
        ww = max(1, math.ceil(math.log2(WDEPTH)))

        self.nt       = CSRStorage(aw,    description="tiles per row (KT = K*nt)")
        self.m_rows   = CSRStorage(mw,    description="output rows")
        self.x_waddr  = CSRStorage(aw,    description="activation BRAM write address")
        self.x_wdata  = CSRStorage(8 * K, description="activation word: K int8 (set before x_we)")
        self.x_we     = CSRStorage(1,     description="commit x_wdata into x_mem[x_waddr]")
        self.wm_waddr = CSRStorage(ww,    description="weight-mem write address (tile index)")
        self.wm_wdata = CSRStorage(2 * K, description="weight tile: K 2-bit codes (set before wm_we)")
        self.wm_we    = CSRStorage(1,     description="commit wm_wdata into w_mem[wm_waddr]")
        self.run      = CSRStorage(1,     description="write 1 to start the timed replay")
        self.rd_addr  = CSRStorage(mw,    description="result row index")
        self.rd_data  = CSRStatus(32,     description="y[rd_addr] (signed int32)")
        self.status   = CSRStatus(1,      description="bit0 = done")
        self.cyc      = CSRStatus(32,     description="measured compute cycles (run -> done)")

        done = Signal()
        rd_data = Signal(32)
        cyc = Signal(32)
        self.comb += [
            self.status.status.eq(done),
            self.rd_data.status.eq(rd_data),
            self.cyc.status.eq(cyc),
        ]

        self.specials += Instance(
            "ternary_gemv_bench",
            p_K=K, p_NT_MAX=NT_MAX, p_M_MAX=M_MAX, p_WDEPTH=WDEPTH,
            i_clk=ClockSignal("sys"),
            i_rst_n=~ResetSignal("sys"),
            i_nt=self.nt.storage,
            i_m_rows=self.m_rows.storage,
            i_x_we=self.x_we.re,
            i_x_waddr=self.x_waddr.storage,
            i_x_wdata=self.x_wdata.storage,
            i_wm_we=self.wm_we.re,
            i_wm_waddr=self.wm_waddr.storage,
            i_wm_wdata=self.wm_wdata.storage,
            i_run=self.run.re,
            i_rd_addr=self.rd_addr.storage,
            o_rd_data=rd_data,
            o_done=done,
            o_cycle_count=cyc,
        )
        for f in RTL_FILES:
            platform.add_source(f"{rtl_dir}/{f}")
