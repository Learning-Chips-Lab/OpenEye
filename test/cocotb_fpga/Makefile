# ==============================================================================
# Simulation Configuration
# ==============================================================================
#SIM ?= icarus
SIM ?= verilator
TOPLEVEL_LANG ?= verilog
WAVES ?= 1

# ==============================================================================
# Simulator-Specific Flags
# ==============================================================================
ifeq ($(SIM),icarus)
    COMPILE_ARGS += -g2012 -DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster

    ifeq ($(WAVES),1)
        unexport WAVES
        unexport COCOTB_TRACE
        unexport IVERILOG_DUMPER
        unexport EXTRA_ARGS
        unexport SIM_ARGS
    endif

else ifeq ($(SIM),verilator)
    COMPILE_ARGS += -Wall -Wno-fatal
    EXTRA_ARGS += --trace-fst --trace-structs -O3 --x-assign fast --x-initial fast \
                  -DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster

    ifeq ($(WAVES),1)
        export COCOTB_TRACE=1
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
# Sources & Paths
# ==============================================================================
HDL_PATH = $(abspath $(PWD)/../../hdl)
VH_PATH  = $(abspath $(PWD))
SIM_ARGS += -v

VERILOG_SOURCES += $(HDL_PATH)/OpenEye_FPGA.v
VERILOG_SOURCES += $(HDL_PATH)/psum_pipeline.v
VERILOG_SOURCES += $(HDL_PATH)/dma_storage.v
VERILOG_SOURCES += $(HDL_PATH)/varlenFIFO.v
VERILOG_SOURCES += $(HDL_PATH)/OpenEye_Parallel.v
VERILOG_SOURCES += $(HDL_PATH)/OpenEye_Cluster.v
VERILOG_SOURCES += $(HDL_PATH)/GLB_cluster.v
VERILOG_SOURCES += $(HDL_PATH)/router_iact.v
VERILOG_SOURCES += $(HDL_PATH)/router_wght.v
VERILOG_SOURCES += $(HDL_PATH)/router_psum.v
VERILOG_SOURCES += $(HDL_PATH)/PE_cluster.v
VERILOG_SOURCES += $(HDL_PATH)/PE.v
VERILOG_SOURCES += $(HDL_PATH)/dsp_unit.v
VERILOG_SOURCES += $(HDL_PATH)/adder.v
VERILOG_SOURCES += $(HDL_PATH)/adder_tree.v
VERILOG_SOURCES += $(HDL_PATH)/delay_cluster.v
VERILOG_SOURCES += $(HDL_PATH)/data_pipeline.v
VERILOG_SOURCES += $(HDL_PATH)/data_pipeline_iact.v
VERILOG_SOURCES += $(HDL_PATH)/data_pipeline_wght.v
VERILOG_SOURCES += $(HDL_PATH)/multiplier.v
VERILOG_SOURCES += $(HDL_PATH)/mux2.v
VERILOG_SOURCES += $(HDL_PATH)/demux2.v
VERILOG_SOURCES += $(HDL_PATH)/mux_iact.v
VERILOG_SOURCES += $(HDL_PATH)/SPad_DP.v
VERILOG_SOURCES += $(HDL_PATH)/SPad_DP_RW.v
VERILOG_SOURCES += $(HDL_PATH)/SPad_SP.v
VERILOG_SOURCES += $(HDL_PATH)/quantization_unit.v
VERILOG_SOURCES += $(HDL_PATH)/RST_SYNC.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_DP.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_DP_RW.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_SP.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_DP_generic.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_DP_RW_generic.v
VERILOG_SOURCES += $(HDL_PATH)/ring_buffer_pipelined.v
VERILOG_SOURCES += $(HDL_PATH)/RAM_SP_generic.v
VERILOG_SOURCES += $(HDL_PATH)/iact_stream_constructor.v
VERILOG_SOURCES += $(HDL_PATH)/router_configurator.v

COMPILE_ARGS += -I$(HDL_PATH)/include

# ==============================================================================
# Cocotb & Timing Configuration
# ==============================================================================
TOPLEVEL = OpenEye_FPGA
export TOPLEVEL = OpenEye_FPGA

MODULE = OpenEye_FPGA_tb
TESTCASE = start_test_fpga

export CLOCK_LEN=10
export CLOCK_UNIT=ns
export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps
export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps

export LOGGER_LEVEL=10    # Enables Logger Debug Mode

# ==============================================================================
# Workload / Layer Configuration
# ==============================================================================
#export LAYER=GEMM
#export LAYER=Residual_Block
#export LAYER=Conv_FC
#export LAYER=FC
#export LAYER=MNIST
#export LAYER=VGG-19
#export LAYER=Convolution_Stack
export LAYER=Conv_Pool_Conv
#export LAYER=Convolution_Single
#export LAYER=Depthwise_Convolution
#export LAYER=MLP
#export LAYER=MobileNetV1
#export LAYER=MaxPooling
#export LAYER=AveragePooling
#export LAYER=Concat_Block

export NUM_FILTERS=16
export KERNEL_SIZE_X=3
export KERNEL_SIZE_Y=3
export INPUT_SIZE_X=16
export INPUT_SIZE_Y=2
export STRIDE=1
export INPUT_CHANNELS=4
export OUTPUT_SIZE=64
export USE_RANDOM_VALUES=1
export USE_SPARSE_IACTS=0
export USE_SPARSE_WEIGHTS=0
export ONLY_FILES=0

# ==============================================================================
# Hardware Architecture Parameters
# ==============================================================================
export PARALLEL_MACS=1
export SPARSITY_EN=0

export DATA_IACT_BITWIDTH=8
export DATA_PSUM_BITWIDTH=32
export DATA_WGHT_BITWIDTH=8
export DATA_IACT_OVERHEAD=4
export DATA_WGHT_IGNORE_ZEROS=4

export TRANS_BITWIDTH_IACT=16
export TRANS_BITWIDTH_WGHT=8
export TRANS_BITWIDTH_PSUM=32

export NUM_GLB_IACT=1
export NUM_GLB_WGHT=3
export NUM_GLB_PSUM=4

export CLUSTER_COLUMNS=1
export CLUSTER_ROWS=1

export IACT_RAM_CELLS=4

export IACT_PER_PE=16
export PE_IACT_ADDR_ADDR=9
export WGHT_PER_PE=96
export PE_WGHT_ADDR_ADDR=16
export PSUM_PER_PE=32

export BRANCHES=2
export BUFFER_WIDTH_IACT=23
export BUFFER_WIDTH_PSUM=24
export BUFFER_WIDTH_WGHT=24
export QUANT_AMOUNT=1024
export TRANS_WORDS=4
export DMA_BITWIDTH=32
export CYCLE_WIDTH=32
export LIMIT_SIZE_IACT=18
export LIMIT_SIZE_WGHT=12
export LIMIT_SIZE_PSUM=16
export CHANNEL_CYCLE_WIDTH=12

export STRIDE_ENABLED=1
export DEPTHWISE_ENABLED=0

export

# ==============================================================================
# Targets
# ==============================================================================
.PHONY: gen_rtl gen_files compile_only run gtkwave clean

gen_rtl:
	@python -m open_eye.generator . ../../hdl/include ../../hdl
	@python -m open_eye.vh_file_creator

# Phase 1 standalone: Generate all reference data & DMA streams without RTL simulation
gen_files:
	@ONLY_FILES=1 python -c "import OpenEye_FPGA_tb as tb; tb.run_precompute_only()"

# Compile RTL once without running simulation (useful for parallel pytest workers)
compile_only: gen_rtl
	@SIM=$(SIM) make -f $(shell cocotb-config --makefiles)/Makefile.sim build

run: gen_rtl
	@make -f $(shell cocotb-config --makefiles)/Makefile.sim

gtkwave:
	gtkwave $(PWD)/sim_build/OpenEye_FPGA.fst

clean:
	rm -rf sim_build demo results.xml