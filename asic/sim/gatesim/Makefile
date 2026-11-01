# Gate-level cocotb regression for the synthesized OpenEye netlist.
#
# Simulates the yosys-mapped netlist (OpenEye_Asic.nl.v) with PDK functional
# models and the same functional test as the RTL regression, to verify that
# the synthesized netlist behaves like the RTL.
#
# Run inside the iic-osic-tools container:
#   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye/asic/sim/gatesim && make run NETLIST=<path-to-OpenEye_Asic.nl.v>"

SIM ?= icarus
TOPLEVEL_LANG ?= verilog
WAVEFORM = off

REPO   = $(abspath $(PWD)/../../..)
OPENEYE_PDK = /foss/pdks/ihp-sg13g2
NETLIST ?= $(REPO)/asic/runs/openeye_p2_9/design/runs/p2_9/06-yosys-synthesis/OpenEye_Asic.nl.v

SRAM_DIR = $(OPENEYE_PDK)/libs.ref/sg13g2_sram/verilog

# stdcell source: PDK original by default (iverilog handles the specify
# blocks and UDP primitives natively). Pass STDC_VLT=sim_build_vlt/sg13g2_stdcell_vlt.v
# to use the generated copy with renamed instances + stripped specify
# blocks (asic/scripts/pdk_specify_strip.py) — functionally identical
# (all PDK specify delays are 0.0) and much faster in Icarus, which
# otherwise schedules a delay event per gate per transition (~200 ns/min
# on the old machine).
STDC_VLT ?= $(OPENEYE_PDK)/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v

VERILOG_SOURCES += $(PWD)/OpenEye_Asic_gatesim_top.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_core_behavioral_bm_bist.v
VERILOG_SOURCES += $(NETLIST)
VERILOG_SOURCES += $(STDC_VLT)
VERILOG_SOURCES += $(OPENEYE_PDK)/libs.ref/sg13g2_stdcell/verilog/sg13g2_udp.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_1024x64_c2_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_1024x16_c2_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_1024x8_c2_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_2048x64_c2_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_2048x32_c2_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_4096x16_c3_bm_bist.v
VERILOG_SOURCES += $(SRAM_DIR)/RM_IHPSG13_1P_4096x8_c3_bm_bist.v
VERILOG_SOURCES += $(OPENEYE_PDK)/libs.ref/sg13g2_io/verilog/sg13g2_io.v

COMPILE_ARGS += -DFUNCTIONAL
COMPILE_ARGS += -Wall

TOPLEVEL = OpenEye_Asic_gatesim_top
export TOPLEVEL = OpenEye_Asic_gatesim_top

MODULE = OpenEye_FPGA_tb
TESTCASE = start_test_fpga

export CLOCK_LEN=10
export CLOCK_UNIT=ns
export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps
export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps
export LOGGER_LEVEL=20

# Functional test (same as test/cocotb_fpga defaults)
export LAYER=Convolution_Single
export NUM_FILTERS=16
export KERNEL_SIZE_X=3
export KERNEL_SIZE_Y=3
export INPUT_SIZE_X=16
export INPUT_SIZE_Y=1
export STRIDE=1
export INPUT_CHANNELS=4
export OUTPUT_SIZE=64
export USE_RANDOM_VALUES=1
export ONLY_FILES=0

# ASIC target configuration (must match the parameters baked into the
# netlist, see asic/target.sh)
export CLUSTER_COLUMNS=2
export CLUSTER_ROWS=2
export NUM_GLB_IACT=1
export NUM_GLB_PSUM=4
export NUM_GLB_WGHT=3
export IACT_RAM_CELLS=4
export BUFFER_WIDTH=10
export BUFFER_WIDTH_WGHT=11
export BUFFER_WIDTH_PSUM=12
export BRANCHES=2
export QUANT_AMOUNT=32
export TRANS_WORDS=8
export STRIDE_ENABLED=0
export DMA_BITWIDTH=64
export PARALLEL_MACS=2
export SPARSITY_EN=1
export DATA_PSUM_BITWIDTH=20
export IACT_SC_ADDRWIDTH=10
export TRANS_BITWIDTH_IACT=24
export TRANS_BITWIDTH_WGHT=24
export TRANS_BITWIDTH_PSUM=20
export USE_SPARSE_IACTS=1
export USE_SPARSE_WEIGHTS=1

export

run:
	rm -rf sim_build
	PYTHONPATH="$(REPO)/src:$(REPO)/test/cocotb_fpga" \
		make -f $(shell cocotb-config --makefiles)/Makefile.sim

clean:
	rm -rf sim_build
