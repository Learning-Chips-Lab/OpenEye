// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// Selects which Processing Element implementation the cluster instantiates.
// Defaults to the sparsity-capable PE.v. Define USE_PE_SIMPLE (e.g.
// `+define+USE_PE_SIMPLE` or cocotb `defines={"USE_PE_SIMPLE": 1}`) to build the
// cluster from the dense-only PE_simple.v. Both modules expose an identical port
// list, so only the instantiated module name changes.
`ifdef USE_PE_SIMPLE
  `define PE_MODULE PE_simple
`else
  `define PE_MODULE PE
`endif

/// Module: PE_cluster
///
/// The PE_cluster module implements a configurable 2D array of Processing Elements (PEs) in the 
/// OpenEye neural network accelerator. It orchestrates data movement and computation across a 
/// grid of PEs while managing communication with global buffers and routing infrastructure.
///
/// Architecture Overview:
/// - Array Organization:
///   * 2D grid of Processing Elements (PEs) arranged in rows and columns
///   * Configurable dimensions through PE_ROWS and PE_COLUMNS parameters
///   * Hierarchical data distribution and collection network
///
/// - Data Flow Mechanisms:
///   * Input Activation (IACT) Distribution:
///     - Multiple global buffer interfaces (NUM_GLB_IACT)
///     - Configurable routing through multiplexers
///     - Individual PE selection capability
///
///   * Weight (WGHT) Distribution:
///     - Row-wise weight distribution
///     - Shared weights across columns
///     - Synchronized weight updates
///
///   * Partial Sum (PSUM) Collection:
///     - Column-wise accumulation path
///     - Dual-mode routing (direct/router)
///     - Configurable output selection
///
/// Key Features:
/// - Flexible PE Array Configuration
/// - Multiple Data Input Sources
/// - Handshake-based Data Transfer
/// - Configurable Routing Paths
/// - Synchronized PE Control
/// - Sparse Data Support
/// - Parallel MAC Operations
///
/// Performance Optimizations:
/// - Efficient Data Distribution
/// - Parallel Processing
/// - Flexible Routing
/// - Pipeline Synchronization
/// - Resource Sharing
///
/// Parameters:
/// Configuration Parameters:
///   IS_TOPLEVEL            - Boolean flag indicating if module is the top level
///   SERIAL                 - Boolean flag enabling serial processing mode
///   TOP_CLUSTER            - Boolean flag indicating topmost cluster in hierarchy
///   PE_ROWS               - Number of processing element rows in the cluster
///   PE_COLUMNS            - Number of processing element columns in the cluster
///   PES                   - Total number of processing elements (PE_ROWS * PE_COLUMNS)
///
/// Processing Configuration:
///   PARALLEL_MACS         - Number of parallel multiply-accumulate operations per PE
///   NUM_GLB_IACT         - Number of input activation global buffer interfaces
///
/// Data Width Parameters:
///   DATA_IACT_BITWIDTH    - Bit width of input activation values
///   DATA_WGHT_BITWIDTH    - Bit width of weight values
///   DATA_PSUM_BITWIDTH    - Bit width of partial sum accumulator
///   TRANS_BITWIDTH_IACT   - Transfer width for input activation interface
///   TRANS_BITWIDTH_WGHT   - Transfer width for weight interface
///   TRANS_BITWIDTH_PSUM   - Transfer width for partial sum interface
///
/// Sparsity Parameters:
///   DATA_IACT_OVERHEAD    - Bits for zero-skipping in input activations
///   DATA_WGHT_IGNORE_ZEROS - Bits for zero-skipping in weights
///
/// Memory Parameters:
///   IACT_ADDR_WORDS      - Input activation address scratchpad depth
///   IACT_DATA_WORDS      - Input activation data scratchpad depth
///   WGHT_ADDR_WORDS      - Weight address scratchpad depth
///   WGHT_DATA_WORDS      - Weight data scratchpad depth
///   PSUM_WORDS           - Partial sum scratchpad depth
///   
/// Ports:
/// System Interface:
///   clk_i                - System clock input
///   rst_ni              - Asynchronous reset (active low)
///   compute_i           - Computation trigger for each PE [PES-1:0]
///   enable_stream_i     - Parameter stream enable
///   data_stream_i       - Configuration parameter data [11:0]
///
/// Routing Control:
///   iact_choose_i       - Input activation routing control
///                        [($clog2(NUM_GLB_IACT+1)*PES)-1:0]
///   psum_choose_i       - Partial sum routing control [PE_COLUMNS-1:0]
///   gemm_mode_i         - GEMM mode enable: when 1, iact_select for each PE is
///                         overridden so that PE row j uses GLB bank j, turning
///                         the cluster into an output-stationary matrix multiplier.
///                         When 0, existing conv behaviour is preserved.
///
/// Systolic GEMM Interface (Approach 3 – requires SYSTOLIC_GEMM_EN=1):
///   iact_pass_data_i    - Horizontal iact pass-through input per row
///                         [DATA_IACT_BITWIDTH*PE_ROWS-1:0]
///   iact_pass_enable_i  - Valid signals for pass-through input [PE_ROWS-1:0]
///   iact_pass_ready_o   - Ready signals for pass-through input [PE_ROWS-1:0]
///   iact_pass_data_o    - Horizontal iact pass-through output per row
///                         [DATA_IACT_BITWIDTH*PE_ROWS-1:0]
///   iact_pass_enable_o  - Valid signals for pass-through output [PE_ROWS-1:0]
///   iact_pass_ready_i   - Ready signals for pass-through output [PE_ROWS-1:0]
///
/// Input Activation Interface:
///   pe_iact_data        - Input activation data bus
///                        [(TRANS_BITWIDTH_IACT*NUM_GLB_IACT)-1:0]
///   pe_iact_enable      - Input activation valid signals [NUM_GLB_IACT-1:0]
///   pe_iact_ready       - Input activation ready signals [NUM_GLB_IACT-1:0]
///
/// Weight Interface:
///   pe_wght_data        - Weight data bus [(TRANS_BITWIDTH_WGHT*PE_ROWS)-1:0]
///   pe_wght_enable      - Weight valid signals [PE_ROWS-1:0]
///   pe_wght_ready       - Weight ready signals [PE_ROWS-1:0]
///
/// Direct Partial Sum Interface:
///   pe_psum_data_i      - Input partial sum bus
///                        [(TRANS_BITWIDTH_PSUM*PE_COLUMNS)-1:0]
///   pe_psum_enable_i    - Input partial sum valid signals [PE_COLUMNS-1:0]
///   pe_psum_ready_i     - Input partial sum ready signals [PE_COLUMNS-1:0]
///   pe_psum_data_o      - Output partial sum bus
///                        [(TRANS_BITWIDTH_PSUM*PE_COLUMNS)-1:0]
///   pe_psum_enable_o    - Output partial sum valid signals [PE_COLUMNS-1:0]
///   pe_psum_ready_o     - Output partial sum ready signals [PE_COLUMNS-1:0]
///
/// Router Partial Sum Interface:
///   pe_router_psum_data_i   - Router input partial sum bus
///                            [(TRANS_BITWIDTH_PSUM*PE_COLUMNS)-1:0]
///   pe_router_psum_enable_i - Router input valid signals [PE_COLUMNS-1:0]
///   pe_router_psum_ready_i  - Router input ready signals [PE_COLUMNS-1:0]
///   pe_router_psum_data_o   - Router output partial sum bus
///                            [(TRANS_BITWIDTH_PSUM*PE_COLUMNS)-1:0]
///   pe_router_psum_enable_o - Router output valid signals [PE_COLUMNS-1:0]
///   pe_router_psum_ready_o  - Router output ready signals [PE_COLUMNS-1:0]
///

module PE_cluster #(
    //Set parameters
    parameter  IS_TOPLEVEL            = 1,
    parameter  SERIAL                 = 1,
    `ifdef USE_INTERNAL_PARAMS_PE_cluster
      parameter  PARALLEL_MACS          = 2,
      parameter  SPARSITY_EN            = 1,  // 1=sparse mode (default), 0=dense mode
    `else
      `include "parameters_PE_cluster.vh"
    `endif
    parameter  TOP_CLUSTER            = 1,
    // Approach 3: set to 1 to add horizontal iact pass-through ports for systolic GEMM
    parameter  SYSTOLIC_GEMM_EN       = 0,
    parameter  DATA_IACT_BITWIDTH     = 8,
    parameter  DATA_PSUM_BITWIDTH     = 20,
    parameter  DATA_WGHT_BITWIDTH     = 8,
    parameter  DATA_IACT_OVERHEAD     = 4,
    parameter  DATA_WGHT_OVERHEAD     = 4,
    parameter  DATA_WGHT_IGNORE_ZEROS = 4,
    // Transfer widths hold PARALLEL_MACS words of the payload plus, in sparse
    // mode, the zero-skip overhead bits. Fixed defaults only matched the dense
    // single-MAC build and truncated every other configuration.
    parameter  TRANS_BITWIDTH_IACT    = PARALLEL_MACS * (DATA_IACT_BITWIDTH + (SPARSITY_EN ? DATA_IACT_OVERHEAD : 0)),
    parameter  TRANS_BITWIDTH_WGHT    = PARALLEL_MACS * (DATA_WGHT_BITWIDTH + (SPARSITY_EN ? DATA_WGHT_IGNORE_ZEROS : 0)),
    parameter  TRANS_BITWIDTH_PSUM    = 20,
    parameter  NUM_GLB_IACT           = 3,
    parameter  IACT_ADDR_WORDS        = 9,
    parameter  IACT_DATA_WORDS        = 16,
    parameter  WGHT_ADDR_WORDS        = 16,
    parameter  WGHT_DATA_WORDS        = 192,
    parameter  PSUM_WORDS             = 32,
    parameter  PE_ROWS                = 3,
    parameter  PE_COLUMNS             = 4,
    localparam PES                    = PE_ROWS * PE_COLUMNS

) (
    input                                         clk_i,
    input                                         rst_ni,
    input [       $clog2(NUM_GLB_IACT+1)*PES-1:0] iact_choose_i,
    input [                       PE_COLUMNS-1:0] psum_choose_i,
    input [                              PES-1:0] compute_i,
    // Approach 2: GEMM mode — row j uses GLB bank j for iact instead of iact_choose_i
    input                                         gemm_mode_i,

    input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] pe_iact_data,
    input  [                    NUM_GLB_IACT-1:0] pe_iact_enable,
    output [                    NUM_GLB_IACT-1:0] pe_iact_ready,

    input  [     TRANS_BITWIDTH_WGHT*PE_ROWS-1:0] pe_wght_data,
    input  [                         PE_ROWS-1:0] pe_wght_enable,
    output [                         PE_ROWS-1:0] pe_wght_ready,

    input  [                      PE_COLUMNS-1:0] pe_psum_ready_i,
    input  [  TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0] pe_psum_data_i,
    input  [                      PE_COLUMNS-1:0] pe_psum_enable_i,

    output [                      PE_COLUMNS-1:0] pe_psum_ready_o,
    output [  TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0] pe_psum_data_o,
    output [                      PE_COLUMNS-1:0] pe_psum_enable_o,

    input [                       PE_COLUMNS-1:0] pe_router_psum_ready_i,
    input [   TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0] pe_router_psum_data_i,
    input [                       PE_COLUMNS-1:0] pe_router_psum_enable_i,

    output [                      PE_COLUMNS-1:0] pe_router_psum_ready_o,
    output [  TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0] pe_router_psum_data_o,
    output [                      PE_COLUMNS-1:0] pe_router_psum_enable_o,

    input                                         enable_stream_i,
    input  [                              12-1:0] data_stream_i,

    // Approach 3: horizontal iact pass-through for systolic GEMM (active when SYSTOLIC_GEMM_EN=1)
    input  [   DATA_IACT_BITWIDTH*PE_ROWS-1:0]   iact_pass_data_i,
    input  [                      PE_ROWS-1:0]   iact_pass_enable_i,
    output [                      PE_ROWS-1:0]   iact_pass_ready_o,
    output [   DATA_IACT_BITWIDTH*PE_ROWS-1:0]   iact_pass_data_o,
    output [                      PE_ROWS-1:0]   iact_pass_enable_o,
    input  [                      PE_ROWS-1:0]   iact_pass_ready_i
);

  ///#######################
  ///Reset synchronization
  ///#######################
  wire                                       rst_nw;


  wire [             PE_ROWS*PE_COLUMNS-1:0] wght_ready_temp;
  wire [PE_COLUMNS*PE_ROWS*NUM_GLB_IACT-1:0] iact_ready_temp;

  reg  [                     PE_COLUMNS-1:0] psum_ready_reg;

  // ============================================================================
  // FST Waveform Dump Configuration (for CocoTB simulation)
  // ============================================================================
  `ifndef NO_TRACE
    reg[1023:0] fst_path;
    initial begin
      if (IS_TOPLEVEL) begin
        // Read the path from the command line argument
        if ($value$plusargs("FST_PATH=%s", fst_path)) begin
          $dumpfile(fst_path);
          $dumpvars(0, PE_cluster);
        end else begin
          // Fallback for when the argument is not provided
          $dumpfile("PE_cluster.fst");
          $dumpvars(0, PE_cluster);
        end
      end
    end
  `endif

  always @(posedge clk_i, negedge rst_nw) begin
    if (!rst_nw) begin : reset
      psum_ready_reg <= 0;
    end else begin
      psum_ready_reg <= pe_psum_ready_i;
    end
  end

  genvar i, j, g;
  generate
    if (IS_TOPLEVEL) begin : gen_reset_control
      RST_SYNC rst_sync_pe_cluster (
          .clk_i (clk_i),
          .rst_ni(rst_ni),
          .rst_no(rst_nw)
      );
    end else begin : direct_reset
      assign rst_nw = rst_ni;
    end

    for (i = 0; i < PE_COLUMNS; i = i + 1) begin : gen_X
      for (j = 0; j < PE_ROWS; j = j + 1) begin : gen_Y
        wire [TRANS_BITWIDTH_PSUM-1 : 0] psum_data_i_w;
        wire                             psum_enable_i_w;
        wire                             psum_ready_i_w;
        wire [TRANS_BITWIDTH_PSUM-1 : 0] psum_data_o_w;
        wire                             psum_enable_o_w;
        wire                             psum_ready_o_w;
        wire [         NUM_GLB_IACT-1:0] iact_ready_o_w;
        wire                             wght_ready_o_w;
        // Approach 3: horizontal iact pass-through wires within each row
        wire [DATA_IACT_BITWIDTH-1:0]    iact_pass_data_i_w;
        wire                             iact_pass_enable_i_w;
        wire                             iact_pass_ready_i_w;
        wire [DATA_IACT_BITWIDTH-1:0]    iact_pass_data_o_w;
        wire                             iact_pass_enable_o_w;
        wire                             iact_pass_ready_o_w;
        // Approach 2: in GEMM mode override iact_select so row j → GLB bank j
        wire [$clog2(NUM_GLB_IACT+1)-1:0] iact_sel_w;
        assign iact_sel_w = gemm_mode_i & SYSTOLIC_GEMM_EN
            ? j[$clog2(NUM_GLB_IACT+1)-1:0]
            : iact_choose_i[(i+j*PE_COLUMNS+1)*$clog2(NUM_GLB_IACT+1)-1
                            :(i+j*PE_COLUMNS)*$clog2(NUM_GLB_IACT+1)];
        `PE_MODULE #(
            .IS_TOPLEVEL           (0),
            .SERIAL                (SERIAL),
            .PARALLEL_MACS         (PARALLEL_MACS),
            .SPARSITY_EN           (SPARSITY_EN),
            .SYSTOLIC_GEMM_EN      (SYSTOLIC_GEMM_EN),
            .DATA_IACT_BITWIDTH    (DATA_IACT_BITWIDTH),
            .DATA_WGHT_BITWIDTH    (DATA_WGHT_BITWIDTH),
            .DATA_PSUM_BITWIDTH    (DATA_PSUM_BITWIDTH),
            .DATA_IACT_OVERHEAD    (DATA_IACT_OVERHEAD),
            .DATA_WGHT_IGNORE_ZEROS(DATA_WGHT_IGNORE_ZEROS),
            .IACT_DATA_ADDR        (IACT_DATA_WORDS),
            .IACT_ADDR_ADDR        (IACT_ADDR_WORDS),
            .WGHT_DATA_ADDR        (WGHT_DATA_WORDS),
            .WGHT_ADDR_ADDR        (WGHT_ADDR_WORDS),
            .PSUM_ADDR             (PSUM_WORDS),
            .TRANS_BITWIDTH_IACT   (TRANS_BITWIDTH_IACT),
            .TRANS_BITWIDTH_WGHT   (TRANS_BITWIDTH_WGHT),
            .NUM_GLB_IACT          (NUM_GLB_IACT)
        ) pe (
            .clk_i(clk_i),
            .rst_ni(rst_nw),
            .iact_select_i(iact_sel_w),
            .compute_i(compute_i[i+j*PE_COLUMNS]),

            .iact_data_i  (pe_iact_data),
            .iact_enable_i(pe_iact_enable),
            .iact_ready_o (iact_ready_o_w),

            .wght_data_i  (pe_wght_data[(j+1)*TRANS_BITWIDTH_WGHT-1:j*TRANS_BITWIDTH_WGHT]),
            .wght_enable_i(pe_wght_enable[j]),
            .wght_ready_o (wght_ready_o_w),

            .psum_data_i    (psum_data_i_w),
            .psum_enable_i  (psum_enable_i_w),
            .psum_ready_o   (psum_ready_o_w),
            .psum_data_o    (psum_data_o_w),
            .psum_enable_o  (psum_enable_o_w),
            .psum_ready_i   (psum_ready_i_w),
            .enable_stream_i(enable_stream_i),
            .data_stream_i  (data_stream_i),
            // Approach 3 systolic pass-through
            .iact_pass_data_i  (iact_pass_data_i_w),
            .iact_pass_enable_i(iact_pass_enable_i_w),
            .iact_pass_ready_o (iact_pass_ready_o_w),
            .iact_pass_data_o  (iact_pass_data_o_w),
            .iact_pass_enable_o(iact_pass_enable_o_w),
            .iact_pass_ready_i (iact_pass_ready_i_w)
        );
      end
    end

    // Approach 3: wire horizontal iact pass-through chains along each row
    // Column 0 of each row connects to the cluster-level input ports.
    // Each subsequent column receives the output of the previous column.
    // The final column's output appears on the cluster-level output ports.
    for (j = 0; j < PE_ROWS; j = j + 1) begin : gen_iact_pass_row
      for (i = 0; i < PE_COLUMNS; i = i + 1) begin : gen_iact_pass_col
        if (SYSTOLIC_GEMM_EN) begin : gen_systolic_wire
          // Broadcast mode: all columns in a row receive the same iact_pass signal
          // directly from the cluster input (no horizontal chaining).
          // This ensures all PEs in a row multiply with the same iact value
          // simultaneously, which is required for correct GEMM operation when
          // weights are shared across columns.
          assign gen_X[i].gen_Y[j].iact_pass_data_i_w   = iact_pass_data_i[(j+1)*DATA_IACT_BITWIDTH-1:j*DATA_IACT_BITWIDTH];
          assign gen_X[i].gen_Y[j].iact_pass_enable_i_w = iact_pass_enable_i[j];
          assign gen_X[i].gen_Y[j].iact_pass_ready_i_w  = iact_pass_ready_i[j];
          if (i == 0) begin : gen_pass_first_col
            assign iact_pass_ready_o[j] = gen_X[i].gen_Y[j].iact_pass_ready_o_w;
          end
          if (i == PE_COLUMNS - 1) begin : gen_pass_last_col
            assign iact_pass_data_o[(j+1)*DATA_IACT_BITWIDTH-1:j*DATA_IACT_BITWIDTH] = gen_X[i].gen_Y[j].iact_pass_data_o_w;
            assign iact_pass_enable_o[j] = gen_X[i].gen_Y[j].iact_pass_enable_o_w;
          end
        end else begin : gen_no_systolic
          // Tie off pass-through wires when SYSTOLIC_GEMM_EN=0
          assign gen_X[i].gen_Y[j].iact_pass_data_i_w   = {DATA_IACT_BITWIDTH{1'b0}};
          assign gen_X[i].gen_Y[j].iact_pass_enable_i_w = 1'b0;
          assign gen_X[i].gen_Y[j].iact_pass_ready_i_w  = 1'b0;
        end
      end
    end
    // When SYSTOLIC_GEMM_EN=0 tie off cluster-level outputs
    if (!SYSTOLIC_GEMM_EN) begin : gen_no_systolic_ports
      assign iact_pass_data_o   = {(DATA_IACT_BITWIDTH*PE_ROWS){1'b0}};
      assign iact_pass_enable_o = {PE_ROWS{1'b0}};
      assign iact_pass_ready_o  = {PE_ROWS{1'b0}};
    end

    // MUX and DEMUX for PSUM Signals
    // MUX PSUM DATA
    genvar k;
    for (k = 0; k < PE_COLUMNS; k = k + 1) begin : psum_data_mux
      wire [TRANS_BITWIDTH_PSUM-1 : 0] out_w;
      mux2 #(
          .DATA_WIDTH(TRANS_BITWIDTH_PSUM)
      ) psum_mux (
          .a_in (pe_router_psum_data_i[(k+1)*TRANS_BITWIDTH_PSUM-1:k*TRANS_BITWIDTH_PSUM]),
          .b_in (pe_psum_data_i[(k+1)*TRANS_BITWIDTH_PSUM-1:k*TRANS_BITWIDTH_PSUM]),
          .sel_i(psum_choose_i[k]),
          .y_o  (out_w)
      );
    end
    /// MUX PSUM ENABLE
    for (k = 0; k < PE_COLUMNS; k = k + 1) begin : psum_enable_mux
      wire out_w;
      mux2 #(
          .DATA_WIDTH(1)
      ) psum_mux (
          .a_in (pe_router_psum_enable_i[k]),
          .b_in (pe_psum_enable_i[k]),
          .sel_i(psum_choose_i[k]),
          .y_o  (out_w)
      );
    end
    /// DEMUX PSUM READY
    for (k = 0; k < PE_COLUMNS; k = k + 1) begin : psum_ready_demux
      wire in_w;
      demux2 #(
          .DATA_WIDTH(1)
      ) psum_demux (
          .a_out(pe_router_psum_ready_o[k]),
          .b_out(pe_psum_ready_o[k]),
          .sel_i(psum_choose_i[k]),
          .i    (in_w)
      );
    end

    /// PE Connections
    for (i = 0; i < PE_COLUMNS; i = i + 1) begin
      for (j = 0; j < PE_ROWS; j = j + 1) begin

        for (g = 0; g < NUM_GLB_IACT; g = g + 1) begin
          assign iact_ready_temp[i+j*PE_COLUMNS+PES*g] = gen_X[i].gen_Y[j].iact_ready_o_w[g];
        end

        assign wght_ready_temp[i+PE_COLUMNS*j] = gen_X[i].gen_Y[j].wght_ready_o_w;

        if (j == 0) begin : gen_first_row
          if (PE_ROWS != 1) begin : gen_single_row
            assign gen_X[i].gen_Y[j+1].psum_ready_i_w = gen_X[i].gen_Y[j].psum_ready_o_w;
          end else begin : gen_multi_row
            assign psum_ready_demux[i].in_w = gen_X[i].gen_Y[j].psum_ready_o_w;
            assign gen_X[i].gen_Y[j].psum_enable_i_w = psum_enable_mux[i].out_w;
            assign gen_X[i].gen_Y[j].psum_data_i_w = psum_data_mux[i].out_w;
          end

          assign gen_X[i].gen_Y[j].psum_ready_i_w = ((TOP_CLUSTER == 1) ? pe_router_psum_ready_i[j] : pe_router_psum_ready_i[j] | psum_ready_reg[j]);
          assign pe_psum_enable_o[i] = gen_X[i].gen_Y[j].psum_enable_o_w;
          assign pe_psum_data_o[(i+1)*TRANS_BITWIDTH_PSUM-1:i*TRANS_BITWIDTH_PSUM]= gen_X[i].gen_Y[j].psum_data_o_w;
          assign pe_router_psum_enable_o[i] = gen_X[i].gen_Y[j].psum_enable_o_w;
          assign pe_router_psum_data_o[(i+1)*TRANS_BITWIDTH_PSUM-1:i*TRANS_BITWIDTH_PSUM] = gen_X[i].gen_Y[j].psum_data_o_w;
        end else begin : gen_not_first_row
          if (j == PE_ROWS - 1) begin : gen_last_row
            assign psum_ready_demux[i].in_w = gen_X[i].gen_Y[j].psum_ready_o_w;
            assign gen_X[i].gen_Y[j-1].psum_enable_i_w = gen_X[i].gen_Y[j].psum_enable_o_w;
            assign gen_X[i].gen_Y[j-1].psum_data_i_w = gen_X[i].gen_Y[j].psum_data_o_w;

            assign gen_X[i].gen_Y[j].psum_enable_i_w = psum_enable_mux[i].out_w;
            assign gen_X[i].gen_Y[j].psum_data_i_w = psum_data_mux[i].out_w;
          end else begin : gen_not_last_row
            assign gen_X[i].gen_Y[j+1].psum_ready_i_w  = gen_X[i].gen_Y[j].psum_ready_o_w;
            assign gen_X[i].gen_Y[j-1].psum_enable_i_w = gen_X[i].gen_Y[j].psum_enable_o_w;
            assign gen_X[i].gen_Y[j-1].psum_data_i_w   = gen_X[i].gen_Y[j].psum_data_o_w;
          end
        end
      end
    end
    ///Connect ready ports
    for (g = 0; g < NUM_GLB_IACT; g = g + 1) begin
      assign pe_iact_ready[g] = ((~iact_ready_temp[(g+1)*PES-1:g*PES]) == {PES{1'b0}});
    end
    for (g = 0; g < PE_ROWS; g = g + 1) begin
      assign pe_wght_ready[g] = ((~wght_ready_temp[((g+1)*PE_COLUMNS)-1:g*PE_COLUMNS]) == {PE_COLUMNS{1'b0}});
    end

  endgenerate

  /// Implementation Notes:
  /// Array Organization:
  ///   - 2D grid arrangement of PEs using nested generate blocks
  ///   - Row-wise weight distribution network
  ///   - Column-wise partial sum accumulation paths
  ///   - Flexible routing through multiplexers and demultiplexers
  ///
  /// Data Distribution:
  ///   - Input Activations:
  ///     * Multiple global buffer interfaces
  ///     * Individual PE selection via iact_choose_i
  ///     * Parallel distribution to all PEs
  ///
  ///   - Weights:
  ///     * Row-wise distribution
  ///     * Shared weights across columns
  ///     * Synchronized weight updates
  ///
  ///   - Partial Sums:
  ///     * Column-wise accumulation
  ///     * Dual routing paths (direct/router)
  ///     * Configurable output selection
  ///
  /// Control Logic:
  ///   - Reset Synchronization:
  ///     * Optional top-level reset handling
  ///     * Clean reset distribution
  ///
  ///   - Handshake Protocol:
  ///     * Ready signal aggregation
  ///     * Enable signal distribution
  ///     * Synchronized data transfer
  ///
  /// Routing Infrastructure:
  ///   - Input Selection:
  ///     * Multiplexers for input activation sources
  ///     * Partial sum input path selection
  ///
  ///   - Output Control:
  ///     * Demultiplexers for partial sum routing
  ///     * Ready signal distribution
  ///
  /// Performance Features:
  ///   - Parallel Processing:
  ///     * Multiple PE operation
  ///     * Concurrent data distribution
  ///     * Synchronized computation
  ///
  ///   - Pipeline Management:
  ///     * Data flow synchronization
  ///     * Handshake coordination
  ///     * Ready signal propagation
  ///
  /// Integration Considerations:
  ///   - Scalability:
  ///     * Configurable array dimensions
  ///     * Adjustable data widths
  ///     * Flexible routing options
  ///
  ///   - Debug Support:
  ///     * Observable control signals
  ///     * Traceable data paths
  ///     * Status monitoring
  ///
  /// Testability Features:
  ///   - Reset Verification
  ///   - Data Path Validation
  ///   - Control Signal Monitoring
  ///   - Performance Measurement
  ///
endmodule
