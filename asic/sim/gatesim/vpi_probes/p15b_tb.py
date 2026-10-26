import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def p15b_test(dut):
    clk = Clock(dut.clk_i, 10, unit="ns")
    clk.start()
    dut.rst_ni.value = 1
    dut.data_dma_i.value = 0
    dut.enable_dma_i.value = 0
    dut.ready_dma_i.value = 0
    await Timer(20, unit="ns")
    dut.data_dma_i.value = 0x123
    # 1) combinational passthrough must update WITHOUT a clock edge
    await Timer(1, unit="ns")
    v = int(dut.data_dma_o.value)
    dut._log.info(f"after write (no edge): data_dma_o={v:#x} (expect [9:1]=0x122)")
    comb_ok = (v & 0x3FE) == 0x122
    # 2) FF path: q captures di[0]=1 at the next posedge; read NBA-safe
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")
    v2 = int(dut.data_dma_o.value)
    dut._log.info(f"after edge: data_dma_o={v2:#x} (expect [9:1]=0x122, [0]=1)")
    ff_ok = (v2 & 0x3FE) == 0x122 and (v2 & 1) == 1
    assert comb_ok, f"COMB propagation failed: {v:#x}"
    assert ff_ok, f"FF path failed: {v2:#x}"
    dut._log.info("P15B_OK")
