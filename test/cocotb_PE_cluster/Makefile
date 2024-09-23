# Makefile

# defaults
#SIM ?= verilator
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# use VHDL_SOURCES for VHDL files
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RST_SYNC.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/multiplier.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/demux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/impl/RAM_DP_RW_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/impl/RAM_SP_generic.v)

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = PE_cluster

# MODULE is the basename of the Python test file
MODULE = PE_cluster_tb

export CLOCK_LEN=10
export CLOCK_UNIT=ns

export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps

export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps


# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
start_simulation:
	touch $(abspath $(PWD)/../../hdl/PE.v)

gtkwave:
	touch $(abspath $(PWD)/../../hdl/PE_cluster.v)
	gtkwave $(PWD)/sim_build/PE_cluster.vcd


