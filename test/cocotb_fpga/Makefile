# ==============================================================================
# Simulation Configuration
# ==============================================================================
SIM ?= icarus
#SIM ?= verilator
TOPLEVEL_LANG ?= verilog
WAVES=1
# ==============================================================================
# Simulator-Spezifische Tracing / Flag Anpassungen
# ==============================================================================
ifeq ($(SIM),icarus)
    # Icarus-spezifische Argumente
    COMPILE_ARGS += -g2012 -DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster
    
    # Waves für Icarus (erzeugt VCD via cocotb)
    ifeq ($(WAVES),1)
        unexport WAVES
        unexport COCOTB_TRACE
        unexport IVERILOG_DUMPER
        unexport EXTRA_ARGS
        unexport SIM_ARGS
        # Cocotb kümmert sich bei WAVES=1 automatisch um den Dump
        # Keine manuellen Dumper-Sources an VERILOG_SOURCES anhängen!
    endif

else ifeq ($(SIM),verilator)
    # Verilator-spezifische Argumente
    COMPILE_ARGS += -Wall -Wno-fatal
    EXTRA_ARGS += --trace-fst --trace-structs -DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster
    
    ifeq ($(WAVES),1)
        SIM_ARGS += --trace-fst
    else
        unexport WAVES
        unexport COCOTB_TRACE
        unexport IVERILOG_DUMPER
        unexport EXTRA_ARGS
        unexport SIM_ARGS
    endif
endif

# ==============================================================================
# Cocotb Include
# ==============================================================================


HDL_PATH = $(abspath $(PWD)/../../hdl)
VH_PATH = $(abspath $(PWD))
SIM_ARGS += -v

# use VHDL_SOURCES for VHDL files
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_FPGA.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/psum_pipeline.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/dma_storage.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/varlenFIFO.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_Parallel.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/OpenEye_Cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/GLB_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_wght.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_psum.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/dsp_unit.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/adder_tree.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/delay_cluster.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/data_pipeline_wght.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/multiplier.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/demux2.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/mux_iact.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/SPad_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/quantization_unit.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RST_SYNC.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_DP_RW_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/ring_buffer_pipelined.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/RAM_SP_generic.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/iact_stream_constructor.v)
VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/router_configurator.v)

COMPILE_ARGS += -I$(abspath $(PWD)/../../hdl/include)

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = OpenEye_FPGA
export TOPLEVEL = OpenEye_FPGA

# MODULE is the basename of the Python test file
MODULE = OpenEye_FPGA_tb
TESTCASE = start_test_fpga
export COCOTB_TRACE=1
export IVERILOG_DUMPER=fst
export CLOCK_LEN=10
export CLOCK_UNIT=ns

export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps

export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps

export LOGGER_LEVEL=10    #Enables Logger Debug Mode

#export LAYER=GEMM
#export LAYER=Residual_Block
#export LAYER=FC
#export LAYER=MNIST
#export LAYER=VGG-19
export LAYER=Convolution_Stack
#export LAYER=Conv_Pool_Conv
#export LAYER=Convolution_Single
#export LAYER=Depthwise_Convolution
#export LAYER=MLP
#export LAYER=MobileNetV1
#export LAYER=MaxPooling
#export LAYER=AveragePooling
export NUM_FILTERS=4
export KERNEL_SIZE_X=3
export KERNEL_SIZE_Y=3
export INPUT_SIZE_X=8
export INPUT_SIZE_Y=2
export STRIDE=1
export INPUT_CHANNELS=4
export OUTPUT_SIZE=10
export USE_RANDOM_VALUES=1
export USE_SPARSE_IACTS=0
export USE_SPARSE_WEIGHTS=0
export ONLY_FILES=0

#Amout of parallel MACs
export PARALLEL_MACS=1
export SPARSITY_EN=0
#Actual Datawidth of Data
export DATA_IACT_BITWIDTH=8
export DATA_PSUM_BITWIDTH=32
export DATA_WGHT_BITWIDTH=8
export DATA_IACT_OVERHEAD=4
export DATA_WGHT_IGNORE_ZEROS=4

#Datawidth for Transportation of Data
export TRANS_BITWIDTH_IACT=16
export TRANS_BITWIDTH_WGHT=8
export TRANS_BITWIDTH_PSUM=32

#Set Number of Router Clusters per Instance
export NUM_GLB_IACT=1
export NUM_GLB_WGHT=3
export NUM_GLB_PSUM=4

#Set number of Clusters
export CLUSTER_COLUMNS=1
export CLUSTER_ROWS=1

#Amoutn of Iact RAM STORAGES
export IACT_RAM_CELLS=4

#Number of Words per PE
export IACT_PER_PE=16
export PE_IACT_ADDR_ADDR=9
export WGHT_PER_PE=96
export PE_WGHT_ADDR_ADDR=16
export PSUM_PER_PE=32

export BRANCHES=1
export BUFFER_WIDTH_IACT = 23
export BUFFER_WIDTH_PSUM = 24
export BUFFER_WIDTH_WGHT = 24
export QUANT_AMOUNT=512
export TRANS_WORDS = 4
export DMA_BITWIDTH = 32
export CYCLE_WIDTH = 32
export LIMIT_SIZE_IACT = 18
export LIMIT_SIZE_WGHT = 12
export LIMIT_SIZE_PSUM = 12
export CHANNEL_CYCLE_WIDTH = 12

export STRIDE_ENABLED = 1
export DEPTHWISE_ENABLED = 1

export
run:
	@python -m open_eye.generator . ../../hdl/include ../../hdl
	@python -m open_eye.vh_file_creator
	@make -f $(shell cocotb-config --makefiles)/Makefile.sim

gtkwave:
	touch $($(HDL_PATH)OpenEye_FPGA.v)
	gtkwave $(PWD)/sim_build/OpenEye_FPGA.vcd

clean:
	rm -rf sim_build
