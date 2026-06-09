"""cocotb test for rtl/ffn_glue_unit.sv — the FFN inter-projection glue on the fabric.
Bit-exact vs models/ffn_glue_unit_ref.glue_unit_int (h_q[f] int8 + amaxN), reports cycles."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

from ffn_glue_unit_ref import glue_unit_int, quantize_weight_fixed

F = 512                     # channels (sim); the real layer is 6912 (~2 cyc/channel)
INTER = 6912
HOST_FFN = 2_580_000        # measured host FFN glue cyc/layer (glue_measured.md)


def m32(x):
    return int(x) & 0xFFFFFFFF


@cocotb.test()
async def ffn_glue_unit(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.we.value = 0
    await Timer(40, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = np.random.default_rng(7)
    gate_int = rng.integers(-6000, 6000, F)
    up_int = rng.integers(-6000, 6000, F)
    norm_w = rng.standard_normal(F) * 0.1 + 1.0
    w_q, _ = quantize_weight_fixed(norm_w, 16)
    hq_exp, amaxN_exp, R, recip = glue_unit_int(gate_int, up_int, w_q)

    # load gate/up/w
    for f in range(F):
        dut.waddr.value = f
        dut.gate_wdata.value = m32(gate_int[f])
        dut.up_wdata.value = m32(up_int[f])
        dut.w_wdata.value = int(w_q[f]) & 0xFFFF
        dut.we.value = 1
        await RisingEdge(dut.clk)
    dut.we.value = 0

    dut.f_count.value = F
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(2_000_000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "timeout waiting for done"

    cyc = int(dut.cycle_count.value)
    got_amax = int(dut.amaxN.value)
    assert got_amax == int(amaxN_exp), f"amaxN {got_amax} != oracle {int(amaxN_exp)}"

    bad = 0
    for f in range(F):
        dut.rd_addr.value = f
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        got = dut.rd_data.value.signed_integer
        if got != int(hq_exp[f]):
            if bad < 8:
                dut._log.info(f"h_q[{f}] = {got}  exp {int(hq_exp[f])}")
            bad += 1

    per_ch = cyc / F
    layer = int(per_ch * INTER)
    dut._log.info(f"FFN_GLUE_UNIT cycles={cyc} (F={F}, {per_ch:.2f} cyc/ch); "
                  f"~/layer(INTER={INTER})={layer} vs host {HOST_FFN} -> {HOST_FFN // max(layer,1)}x; "
                  f"amaxN~2^{int(amaxN_exp).bit_length()} recip~2^{int(recip).bit_length()}")
    assert bad == 0, f"{bad}/{F} h_q mismatches"
    dut._log.info("FFN_GLUE_UNIT_PASS")
