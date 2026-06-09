"""LiteX CSR peripheral wrapping rtl/attention_unit.sv — on-fabric attention on silicon.

The CPU loads q[D], the KV cache (T keys x D, k+v together), and the Q15 exp LUT via CSRs,
sets t_keys/score_shift, pulses `start`; the unit computes scores -> shift+LUT softmax -> a@V
and the hardware `cycle_count` captures the latency. The CPU reads num[d] + sum_e (ctx=num/sum
is a trivial host step). T_MAX=64 keeps BRAM within budget alongside the SoC.
"""
import math

from migen import Module, Signal, ClockSignal, ResetSignal, Instance
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

RTL_FILES = ["attention_unit.sv"]


class AttentionUnit(Module, AutoCSR):
    def __init__(self, platform, rtl_dir, D=128, T_MAX=64, EXP_N=4096):
        dw = max(1, math.ceil(math.log2(D)))
        tw = max(1, math.ceil(math.log2(T_MAX)))
        ew = max(1, math.ceil(math.log2(EXP_N)))

        self.t_keys      = CSRStorage(tw + 1, description="number of keys (1..T_MAX)")
        self.score_shift = CSRStorage(6,      description="softmax logit right-shift")
        self.q_waddr     = CSRStorage(dw,     description="q BRAM write address")
        self.q_wdata     = CSRStorage(16,     description="q word (int16)")
        self.q_we        = CSRStorage(1,      description="commit q_wdata")
        self.kv_waddr    = CSRStorage(tw + dw, description="KV write address (j*D+d)")
        self.k_wdata     = CSRStorage(16,     description="k word (int16)")
        self.v_wdata     = CSRStorage(16,     description="v word (int16)")
        self.kv_we       = CSRStorage(1,      description="commit k_wdata+v_wdata")
        self.lut_waddr   = CSRStorage(ew,     description="exp LUT write address")
        self.lut_wdata   = CSRStorage(16,     description="exp LUT word (Q15)")
        self.lut_we      = CSRStorage(1,      description="commit lut_wdata")
        self.start       = CSRStorage(1,      description="write 1 to run")
        self.rd_addr     = CSRStorage(dw,     description="num[] read index")
        self.rd_data     = CSRStatus(48,      description="num[rd_addr] (signed 48)")
        self.sum_e       = CSRStatus(32,      description="sum of exp weights")
        self.status      = CSRStatus(1,       description="bit0 = done")
        self.cyc         = CSRStatus(32,      description="measured cycles (start -> done)")

        done = Signal(); rd_data = Signal(48); sum_e = Signal(32); cyc = Signal(32)
        self.comb += [
            self.status.status.eq(done),
            self.rd_data.status.eq(rd_data),
            self.sum_e.status.eq(sum_e),
            self.cyc.status.eq(cyc),
        ]
        self.specials += Instance(
            "attention_unit",
            p_D=D, p_T_MAX=T_MAX, p_EXP_N=EXP_N,
            i_clk=ClockSignal("sys"),
            i_rst_n=~ResetSignal("sys"),
            i_t_keys=self.t_keys.storage,
            i_score_shift=self.score_shift.storage,
            i_q_we=self.q_we.re, i_q_waddr=self.q_waddr.storage, i_q_wdata=self.q_wdata.storage,
            i_kv_we=self.kv_we.re, i_kv_waddr=self.kv_waddr.storage,
            i_k_wdata=self.k_wdata.storage, i_v_wdata=self.v_wdata.storage,
            i_lut_we=self.lut_we.re, i_lut_waddr=self.lut_waddr.storage,
            i_lut_wdata=self.lut_wdata.storage,
            i_start=self.start.re,
            i_rd_addr=self.rd_addr.storage,
            o_done=done, o_rd_data=rd_data, o_sum_e=sum_e, o_cycle_count=cyc,
        )
        for f in RTL_FILES:
            platform.add_source(f"{rtl_dir}/{f}")
