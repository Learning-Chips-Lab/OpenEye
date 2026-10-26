import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def p15d_test(dut):
    clk = Clock(dut.clk_i, 10, unit="ns")
    clk.start()
    dut.rst_ni.value = 1
    dut.data_dma_i.value = 0
    dut.enable_dma_i.value = 0
    dut.ready_dma_i.value = 0
    await Timer(20, unit="ns")
    dut.data_dma_i.value = 0x123
    # 1) combinational passthrough must update WITHOUT a clock edge
    #    (and2 chain on real PDK cells + IOPadIn)
    await Timer(1, unit="ns")
    v = int(dut.data_dma_o.value)
    dut._log.info(f"after write (no edge): data_dma_o={v:#x} (expect bits[9:1]=0x122)")
    comb_ok = (v & 0x3FE) == 0x122
    # 2) FF path: the FIRST dfrbpq (ffq0) captures data_dma_i[0]=1 at the
    #    next posedge and is tapped out on ffq0_o (data_dma_o[0] is the
    #    END of the NF-deep chain, so it is NOT the check target).
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")
    v2 = int(dut.data_dma_o.value)
    f = int(dut.ffq0_o.value)
    dut._log.info(f"after edge: data_dma_o={v2:#x} (expect bits[9:1]=0x122), "
                 f"ffq0_o={f} (expect 1)")
    ff_ok = (v2 & 0x3FE) == 0x122 and f == 1
    assert comb_ok, f"COMB propagation failed: {v:#x}"
    assert ff_ok, f"FF path failed: data_dma_o={v2:#x} ffq0_o={f}"
    dut._log.info("P15D_OK")
