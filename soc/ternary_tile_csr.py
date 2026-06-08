"""LiteX CSR peripheral wrapping the ternary_tile engine.

Exposes the multiply-free ternary GEMV engine to the SoC's CPU as memory-mapped
CSRs: load the activation vector (`x`), stream dense base-3 weight bytes (`wbyte`,
one per write), pulse `ctl` to start, poll `status.done`, then read y[m] back via
`rd_addr`/`rd_data`. Lets the RISC-V core run a real GEMV on the fabric engine and
read the result — the on-board compute proof, with weights resident in DDR3.
"""
import math

from migen import Module, Signal, ClockSignal, ResetSignal, Instance
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

RTL_FILES = [
    "ternary_dot_pipe.sv", "ternary_gemv_pipe.sv",
    "ternary_unpack5.sv", "weight_feed.sv", "ternary_tile.sv",
]


class TernaryTile(Module, AutoCSR):
    def __init__(self, platform, rtl_dir, K=10, M=16):
        rw = max(1, math.ceil(math.log2(M)))

        self.x       = CSRStorage(8 * K, description="activation vector: K int8, little-endian")
        self.wbyte   = CSRStorage(8,     description="weight byte; each write pushes one base-3 byte")
        self.ctl     = CSRStorage(1,     description="write 1 to start (latch x, reset counters)")
        self.status  = CSRStatus(1,      description="bit0 = done")
        self.rd_addr = CSRStorage(rw,    description="result row index to read")
        self.rd_data = CSRStatus(32,     description="y[rd_addr] (signed int32)")

        done    = Signal()
        rd_data = Signal(32)
        self.comb += [self.status.status.eq(done), self.rd_data.status.eq(rd_data)]

        self.specials += Instance(
            "ternary_tile",
            p_K=K, p_M=M,
            i_clk=ClockSignal("sys"),
            i_rst_n=~ResetSignal("sys"),
            i_start=self.ctl.re,            # 1-cycle pulse on CSR write
            i_x_flat=self.x.storage,
            i_wbyte=self.wbyte.storage,
            i_wbyte_valid=self.wbyte.re,    # 1-cycle byte_valid pulse per write
            i_rd_addr=self.rd_addr.storage,
            o_rd_data=rd_data,
            o_done=done,
        )
        for f in RTL_FILES:
            platform.add_source(f"{rtl_dir}/{f}")
