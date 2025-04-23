# Makefile

# Icarus Verilog
SIM ?= icarus

# Verilator
#SIM ?= verilator
#EXTRA_ARGS += --trace --trace-fst --trace-structs
#COMPILE_ARGS += --trace --trace-fst --trace-structs
VERILOG_COMPILE_ARGS += -DUSE_EXTERNAL_PARAMS
TOPLEVEL_LANG ?= verilog
WAVEFORM=on

# use VHDL_SOURCES for VHDL files
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_FPGA.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/varlenFIFO.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_Parallel.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_Cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/GLB_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/af_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/bano_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_wght.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_psum.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/delay_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/multiplier.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/demux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RST_SYNC.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/RAM_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/impl/RAM_DP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/impl/RAM_DP_RW_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/memory/impl/RAM_SP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/iact_stream_constructor.v)


# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = OpenEye_FPGA

# MODULE is the basename of the Python test file
MODULE = OpenEye_FPGA_tb
export COCOTB_TRACE=1
export IVERILOG_DUMPER=fst
export CLOCK_LEN=10
export CLOCK_UNIT=ns

export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps

export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps

export LOGGER_LEVEL=10    #Enables Logger Debug Mode

#export LAYER=FC
export LAYER=Convolution
#export LAYER=Depthwise_Convolution
export NUM_FILTERS=8
export KERNEL_SIZE=3
export INPUT_SIZE=16
export OUTPUT_SIZE=8
export STRIDE=1
export INPUT_CHANNELS=4
#Amout of parallel MACs
export PARALLEL_MACS=2
export USE_RANDOM_VALUES=1
export USE_SPARSE_IACTS=0
export USE_SPARSE_WEIGHTS=0
	
#Actual Datawidth of Data
export DATA_IACT_BITWIDTH=8
export DATA_PSUM_BITWIDTH=20
export DATA_WGHT_BITWIDTH=8
export DATA_IACT_OVERHEAD=4
export DATA_WGHT_IGNORE_ZEROS=4

#Datawidth for Transportation of Data
export TRANS_BITWIDTH_IACT=24
export TRANS_BITWIDTH_WGHT=24
export TRANS_BITWIDTH_PSUM=20

#Set Number of Router Clusters per Instance
export NUM_GLB_IACT=3
export NUM_GLB_WGHT=3
export NUM_GLB_PSUM=4

#Set Number of PEelements
export PE_ROWS=3
export PE_COLUMNS=4

#Set number of Cluster rows need at least to be 2 COLUMNS
export CLUSTER_COLUMNS=2
export CLUSTER_ROWS=8

#Number of Words per PE
export IACT_PER_PE=16
export PE_IACT_ADDR_ADDR=9
export WGHT_PER_PE=96
export PE_WGHT_ADDR_ADDR=16
export PSUM_PER_PE=32

# include cocotb's make rules to take care of the simulator setup
export SCRIPT = $(PWD)/../test_utils/vh_file_creator.py

export
run:
	@python $(SCRIPT) && make -f $(shell cocotb-config --makefiles)/Makefile.sim

gtkwave:
	touch $(abspath $(PWD)/../../hdl/OpenEye_FPGA.v)
	gtkwave $(PWD)/sim_build/OpenEye_FPGA.vcd

clean:
	rm -rf sim_build
