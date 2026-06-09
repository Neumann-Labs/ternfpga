"""LiteX CSR peripheral wrapping ternary_gemv_stream — the BRAM-centric streaming GEMV.

Exposes the scalable, real-width ternary GEMV to the CPU as memory-mapped CSRs:
  1. set `nt` (tiles/row, KT=K*nt) and `m_rows` (output rows),
  2. load the activation into the engine BRAM — for each tile word set `x_waddr` then
     write `x_wdata` (each write pulses x_we),
  3. write 1 to `ctl` to start (reset counters, latch dims),
  4. stream weight tiles: each `w_tile` write pushes one K-wide ternary tile (row-major),
  5. poll `status.done`, then read y[m] via `rd_addr` / `rd_data`.

K=8 keeps `x_wdata` a clean 64-bit CSR (>64-bit CSRs have no C accessor — Phase-1 lesson).
"""
import math

from migen import Module, Signal, ClockSignal, ResetSignal, Instance
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

RTL_FILES = ["ternary_dot_pipe.sv", "ternary_gemv_stream.sv"]


class TernaryGemvStream(Module, AutoCSR):
    def __init__(self, platform, rtl_dir, K=8, NT_MAX=64, M_MAX=64):
        aw = max(1, math.ceil(math.log2(NT_MAX)))
        mw = max(1, math.ceil(math.log2(M_MAX)))

        self.nt      = CSRStorage(aw,     description="tiles per row (KT = K*nt)")
        self.m_rows  = CSRStorage(mw,     description="output rows")
        self.x_waddr = CSRStorage(aw,     description="activation BRAM write address (tile index)")
        self.x_wdata = CSRStorage(8 * K,  description="activation word: K int8 (set before pulsing x_we)")
        self.x_we    = CSRStorage(1,      description="write 1 to commit x_wdata into x_mem[x_waddr]")
        self.ctl     = CSRStorage(1,      description="write 1 to start (reset counters, latch dims)")
        self.w_tile  = CSRStorage(2 * K,  description="weight tile: K 2-bit codes; each write pushes one")
        self.rd_addr = CSRStorage(mw,     description="result row index to read")
        self.rd_data = CSRStatus(32,      description="y[rd_addr] (signed int32)")
        self.status  = CSRStatus(1,       description="bit0 = done")

        done    = Signal()
        rd_data = Signal(32)
        self.comb += [self.status.status.eq(done), self.rd_data.status.eq(rd_data)]

        self.specials += Instance(
            "ternary_gemv_stream",
            p_K=K, p_NT_MAX=NT_MAX, p_M_MAX=M_MAX,
            i_clk=ClockSignal("sys"),
            i_rst_n=~ResetSignal("sys"),
            i_nt=self.nt.storage,
            i_m_rows=self.m_rows.storage,
            i_x_we=self.x_we.re,            # 1-cycle commit pulse (data stable before)
            i_x_waddr=self.x_waddr.storage,
            i_x_wdata=self.x_wdata.storage,
            i_start=self.ctl.re,            # 1-cycle start pulse
            i_w_valid=self.w_tile.re,       # 1-cycle tile-valid pulse per write
            i_w_tile=self.w_tile.storage,
            i_rd_addr=self.rd_addr.storage,
            o_rd_data=rd_data,
            o_done=done,
        )
        for f in RTL_FILES:
            platform.add_source(f"{rtl_dir}/{f}")
