# Makefile

# defaults
#SIM ?= verilator
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
#EXTRA_ARGS += --trace-fst --trace-structs

HDL_PATH = $(abspath $(PWD)/../../hdl)
VH_PATH = $(abspath $(PWD))
# use VHDL_SOURCES for VHDL files
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder_tree.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_wght.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/dsp_unit.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/multiplier.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP_generic.v)

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = PE
export TOPLEVEL = PE

# MODULE is the basename of the Python test file
MODULE = PE_tb
export IVERILOG_DUMPER=fst

export CLOCK_LEN=10
export CLOCK_UNIT=ns

export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps

export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps

export LOGGER_LEVEL=0    #Enables Logger Debug Mode

export B=2
export C0=4
export M0=4
export SEED = 0
export PARALLEL_MACS = 2
export SPARSITY_EN=1
export SPARSE_IACT=0
export SPARSE_WGHT=30

export
run:
	@python -m open_eye.vh_file_creator
	@make -f $(shell cocotb-config --makefiles)/Makefile.sim