"""LiteX CSR peripheral wrapping rtl/ffn_glue_unit.sv — on-fabric FFN glue on silicon.

The CPU loads gate_int/up_int/w_q per channel via CSRs, sets f_count, pulses `start`; the unit
computes N=relu(gate)^2*up*w_q, max|N|, and the int8 requant h_q=round(N*127/max|N|), and the
hardware `cycle_count` captures the latency. The CPU reads h_q[f] (int8, the down_proj input) +
max|N| (for the down_proj output dequant). FFN-glue 20 BRAM + the SoC's ~27 = 47 < 50 -> fits a 35T.
"""
import math

from migen import Module, Signal, ClockSignal, ResetSignal, Instance
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

RTL_FILES = ["ffn_glue_unit.sv"]


class FfnGlueUnit(Module, AutoCSR):
    def __init__(self, platform, rtl_dir, F_MAX=6912):
        fw = max(1, math.ceil(math.log2(F_MAX)))

        self.f_count    = CSRStorage(fw + 1, description="number of channels (<= F_MAX)")
        self.waddr      = CSRStorage(fw,     description="channel write address")
        self.gate_wdata = CSRStorage(32,     description="gate_int word")
        self.up_wdata   = CSRStorage(32,     description="up_int word")
        self.w_wdata    = CSRStorage(16,     description="w_q (fixed-point norm weight)")
        self.we         = CSRStorage(1,      description="commit gate+up+w at waddr")
        self.start      = CSRStorage(1,      description="write 1 to run")
        self.rd_addr    = CSRStorage(fw,     description="h_q[] read index")
        self.rd_data    = CSRStatus(8,       description="h_q[rd_addr] (signed int8)")
        self.amax_lo    = CSRStatus(32,      description="max|N| bits [31:0]")
        self.amax_mid   = CSRStatus(32,      description="max|N| bits [63:32]")
        self.amax_hi    = CSRStatus(32,      description="max|N| bits [95:64]")
        self.status     = CSRStatus(1,       description="bit0 = done")
        self.cyc        = CSRStatus(32,      description="measured cycles (start -> done)")

        done = Signal(); rd_data = Signal(8); amaxN = Signal(96); cyc = Signal(32)
        self.comb += [
            self.status.status.eq(done),
            self.rd_data.status.eq(rd_data),
            self.amax_lo.status.eq(amaxN[0:32]),
            self.amax_mid.status.eq(amaxN[32:64]),
            self.amax_hi.status.eq(amaxN[64:96]),
            self.cyc.status.eq(cyc),
        ]
        self.specials += Instance(
            "ffn_glue_unit",
            p_F_MAX=F_MAX,
            i_clk=ClockSignal("sys"),
            i_rst_n=~ResetSignal("sys"),
            i_f_count=self.f_count.storage,
            i_we=self.we.re, i_waddr=self.waddr.storage,
            i_gate_wdata=self.gate_wdata.storage, i_up_wdata=self.up_wdata.storage,
            i_w_wdata=self.w_wdata.storage,
            i_start=self.start.re,
            i_rd_addr=self.rd_addr.storage,
            o_done=done, o_rd_data=rd_data, o_amaxN=amaxN, o_cycle_count=cyc,
        )
        for f in RTL_FILES:
            platform.add_source(f"{rtl_dir}/{f}")
