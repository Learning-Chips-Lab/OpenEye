# OpenEye ASIC target configuration (IHP SG13G2, first tapeout candidate).
#
# Source this (or import the values) for RTL generation, cocotb regression
# and LibreLane runs. 2x2 clusters = 48 PEs. Buffer depths are aligned to
# the available sg13g2 SRAM macro grid (see asic/README.md + tiling table):
#
#   BUFFER_A   4 x 1024 x 64   -> 4 x RM_IHPSG13_1P_1024x64
#   wght_buf  1 x 2048 x 288   -> 4 x RM_..._2048x64 + 1 x RM_..._2048x32
#   psum_buf  8 x 4096 x 40    -> 8 x (2 x RM_..._4096x16 + 1 x RM_..._4096x8)
#   iact_sc   4 x 1024 x 72    -> 4 x (RM_..._1024x64 + RM_..._1024x8)
#
# PE-internal scratchpads stay standard-cell latches (see Phase-1 notes).
#
# NUM_GLB_IACT=1: the NUM_GLB_IACT=3 iact routing path is broken in the
# current codebase snapshot (cocotb result mismatch, array-size independent,
# see asic/runs/cocotb_1x1_glb3.log). 1 is the verified configuration.

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
export DATAFLOW=row_stationary

# iact_stream_constructor RAM depth: 1024 rows (width 72 = 64+8 only exists
# up to 1024 rows in the sg13g2 SRAM grid).
export IACT_SC_ADDRWIDTH=10
