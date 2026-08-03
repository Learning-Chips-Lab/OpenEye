# Makefile

# defaults
#SIM ?= verilator
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# use VERILOG_SOURCES for Verilog files
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RST_SYNC.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder_tree.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_wght.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/dsp_unit.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/multiplier.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/demux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP_generic.v)

export VERILOG_SOURCES

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = PE_cluster

export TOPLEVEL = PE_cluster

# MODULE is the basename of the Python test file
MODULE = PE_cluster_tb
export MODULE = PE_cluster_tb

export CLOCK_LEN=10
export CLOCK_UNIT=ns

export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps

export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps

export LOGGER_LEVEL=0    #Enables Logger Debug Mode

# Default PE/cluster parameters (can be overridden by caller)
export PARALLEL_MACS = 1
export SPARSITY_EN = 0
export SPARSE_IACT=0
export SPARSE_WGHT=0
export SEED = 42
export IACTSIZE_X=3
export IACTSIZE_Y=4
export WGHTSIZE_X=8
export WGHTSIZE_Y=12
export IVERILOG_DUMPER=fst
COMPILE_ARGS += -I$(abspath $(PWD)/../../hdl/include)
export


# include cocotb's make rules to take care of the simulator setup
export
run:
	@python -m open_eye.vh_file_creator
	@make -f $(shell cocotb-config --makefiles)/Makefile.sim


