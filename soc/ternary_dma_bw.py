#!/usr/bin/env python3
"""Sustained DDR3 read-bandwidth instrument — the roofline behind the energy/token ceiling.

A hardware LiteDRAMDMAReader pumps `length` sequential native-port words from `base`, drains
them as fast as the controller delivers, and a hardware counter measures run cycles. The CPU
then computes:  bandwidth = length * (port.data_width/8) / (cycles / sys_clk_freq).

This is the TRUE roofline (a wide hardware DMA), NOT the BIOS Memspeed (the CPU's slow
wishbone memcpy). Batch-1 LLM decode is bandwidth-bound — it streams every weight once per
token — so this single GB/s number caps full-model tok/s:  tok/s <= GB/s / bytes_per_token.

It is a measurement instrument (a counter over a real DMA): there is no datapath correctness
to simulate — the number only means anything on silicon. The DMA->engine datapath correctness
is simulated separately when the two integrate (Phase-3 #30).
"""
from migen import *
from litex.soc.interconnect.csr import AutoCSR, CSRStorage, CSRStatus

from litedram.frontend.dma import LiteDRAMDMAReader


class TernaryDmaBw(Module, AutoCSR):
    def __init__(self, port, fifo_depth=32):
        self.base    = CSRStorage(32)   # start address, in native DRAM words
        self.length  = CSRStorage(32)   # number of native words to read
        self.start   = CSRStorage()     # write -> .re pulse begins a run
        self.done    = CSRStatus()      # 1 when the run has finished
        self.cycles  = CSRStatus(32)    # measured run cycles (start -> last word)
        self.words   = CSRStatus(32)    # words actually received (sanity == length)
        self.chksum  = CSRStatus(32)    # XOR of low 32 data bits (proves bytes moved)
        self.dwbytes = CSRStatus(16)    # port.data_width in bytes (host computes GB/s)

        # # #

        reader = LiteDRAMDMAReader(port, fifo_depth=fifo_depth, fifo_buffered=True)
        self.submodules += reader

        self.comb += self.dwbytes.status.eq(port.data_width // 8)

        running  = Signal()
        issued   = Signal(32)
        recvd    = Signal(32)
        base_r   = Signal(32)
        length_r = Signal(32)
        cyc      = Signal(32)
        chk      = Signal(32)

        # Issue sequential addresses while running; always drain the source.
        self.comb += [
            reader.sink.valid.eq(running & (issued < length_r)),
            reader.sink.address.eq(base_r + issued),
            reader.source.ready.eq(1),
        ]

        self.sync += [
            If(self.start.re,
                running.eq(1),
                issued.eq(0),
                recvd.eq(0),
                cyc.eq(0),
                chk.eq(0),
                base_r.eq(self.base.storage),
                length_r.eq(self.length.storage),
                self.done.status.eq(0),
            ).Elif(running,
                cyc.eq(cyc + 1),
                If(reader.sink.valid & reader.sink.ready,
                    issued.eq(issued + 1),
                ),
                If(reader.source.valid & reader.source.ready,
                    recvd.eq(recvd + 1),
                    chk.eq(chk ^ reader.source.data[:32]),
                    # completion is gated on a REAL arriving word (the length_r-th).
                    If(recvd == (length_r - 1),
                        running.eq(0),
                        self.done.status.eq(1),
                        self.cycles.status.eq(cyc + 1),
                        self.words.status.eq(recvd + 1),
                        self.chksum.status.eq(chk ^ reader.source.data[:32]),
                    ),
                ),
            ),
        ]
