"""cocotb test for rtl/attention_unit.sv — one query vs a BRAM KV cache, bit-exact vs
models/attn_unit_ref.attn_unit_int (num[d] + sum_e), and reports cycles/query."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

from attn_unit_ref import attn_unit_int, exp_lut, EXP_LSB

D, T = 128, 32


def m16(x):
    return int(x) & 0xFFFF


async def _wr(dut, signal_we, awaddr, addr, adata, data):
    awaddr.value = addr
    adata.value = m16(data)
    signal_we.value = 1
    await RisingEdge(dut.clk)
    signal_we.value = 0


@cocotb.test()
async def attention_unit(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    for s in ("start", "q_we", "kv_we", "lut_we"):
        getattr(dut, s).value = 0
    await Timer(40, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    rng = np.random.default_rng(11)
    q = rng.standard_normal(D) * 0.5
    K = rng.standard_normal((T, D)) * 0.5
    V = rng.standard_normal((T, D)) * 0.5
    qs, ks, vs = (np.abs(a).max() / 32767.0 for a in (q, K, V))
    q_i = np.round(q / qs).astype(np.int64)
    K_i = np.round(K / ks).astype(np.int64)
    V_i = np.round(V / vs).astype(np.int64)
    score_shift = int(round(np.log2(EXP_LSB * np.sqrt(D) / (qs * ks))))
    EL = exp_lut()
    _, _, _, s_exp, num_exp = attn_unit_int(q_i, K_i, V_i, score_shift, EL)

    # load exp LUT
    for i in range(len(EL)):
        dut.lut_waddr.value = i; dut.lut_wdata.value = m16(EL[i]); dut.lut_we.value = 1
        await RisingEdge(dut.clk)
    dut.lut_we.value = 0
    # load q
    for d in range(D):
        dut.q_waddr.value = d; dut.q_wdata.value = m16(q_i[d]); dut.q_we.value = 1
        await RisingEdge(dut.clk)
    dut.q_we.value = 0
    # load KV (k and v at same addr j*D+d)
    for j in range(T):
        for d in range(D):
            dut.kv_waddr.value = j * D + d
            dut.k_wdata.value = m16(K_i[j, d]); dut.v_wdata.value = m16(V_i[j, d])
            dut.kv_we.value = 1
            await RisingEdge(dut.clk)
    dut.kv_we.value = 0

    dut.t_keys.value = T
    dut.score_shift.value = score_shift
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(3_000_000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "timeout waiting for done"

    cyc = int(dut.cycle_count.value)
    got_sum = int(dut.sum_e.value)
    assert got_sum == int(s_exp), f"sum_e {got_sum} != oracle {int(s_exp)}"

    bad = 0
    for d in range(D):
        dut.rd_addr.value = d
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        got = dut.rd_data.value.signed_integer
        if got != int(num_exp[d]):
            if bad < 8:
                dut._log.info(f"num[{d}] = {got}  exp {int(num_exp[d])}")
            bad += 1

    layer = cyc * 20            # ~20 q-heads/layer (BitNet-2B)
    dut._log.info(f"ATTN_UNIT cycles/query={cyc} (T={T} D={D}); ~/layer(x20 heads)={layer} "
                  f"vs host 16.2M -> {16213542 // max(layer,1)}x faster; sum_e={got_sum}")
    assert bad == 0, f"{bad}/{D} num[] mismatches"
    dut._log.info("ATTENTION_UNIT_PASS")
