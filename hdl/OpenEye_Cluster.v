// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_Cluster
///

/// The OpenEye_Cluster contains the complete routing architecture, the required storage and the
/// processing elements. Data is first stored in the GLB (IACT/PSUM) or directly in the PE
/// (WGHT). After computation in the process elements, the bias is loaded from the GLBs and passed
/// to the process elements. After this process the results are collected and the data is passed
/// through the af_cluster and the bano_cluster. From there the results are written back
/// into the PSUM GLB and are ready to be read.
///
/// 
/// Router ports and handshake protocol:
/// Several ports are used for connections - they contain the data, enable and ready signals.
/// These ports are the connection to the other routers. They are divided into 3 categories and
/// 3 bus types; IACT, WGHT and PSUM are the 3 categories, while the handshake protocol is done by
/// the 3 busses. Data, Enable and Ready. When source and destination are ready, source enable
/// starts the data communication
///
/// Parameters:
///   IS_TOPLEVEL            - Decides, wether modul is topmodul or not
///   LEFT_CLUSTER           - Indicates, if cluster is on the left side
///   TOP_CLUSTER            - Decides, wether this cluster is the topmost cluster
///   BOTTOM_CLUSTER         - Indicates, wether this cluster is the topmost cluster
///   DATA_IACT_BITWIDTH     - Width of input activation data
///   DATA_WGHT_BITWIDTH     - Width of weight data
///   DATA_PSUM_BITWIDTH     - Width of partial sum data, used in internal accumulator
///   TRANS_BITWIDTH_IACT    - Width of iact input port 
///   TRANS_BITWIDTH_WGHT    - Width of weight input port
///   TRANS_BITWIDTH_PSUM    - Width of partial sum input port
///   NUM_GLB_IACT           - Number of input activation global buffers
///   NUM_GLB_WGHT           - Number of rows of PEs in cluster (No WGHT GLBs)
///   NUM_GLB_PSUM           - Number of partial sum global buffers
///   PE_ROWS                - Amount of rows of process elements
///   PE_COLUMNS             - Amount of columns of process elements
///   PES                    - Amount of process elements
///   CLUSTER_ROWS           - Amount of rows of clusters
///   CLUSTER_COLUMNS        - Amount of columns of clusters
///   CLUSTERS               - Amount of clusters
///   IACT_PER_PE            - Maximum Iact Words in process element
///   WGHT_PER_PE            - Maximum Wght Words in process element
///   PSUM_PER_PE            - Maximum Psum Words in process element
///   IACT_MEM_ADDR_WORDS    - Number of words in IACT GLB
///   PSUM_MEM_ADDR_WORDS    - Number of words in PSUM GLB
///   IACT_MEM_ADDR_BITS     - Width of words in IACT GLB
///   PSUM_MEM_ADDR_BITS     - Width of words in PSUM GLB
///   BANO_MODES             - Amount of Modes in Batch Normalization
///   AF_MODES               - Amount of Modes in AutoFunction CLuster
/// 
/// Ports:
///   iact_choose_i           - Specify the iact MUX and DEMUX for DATA, ENABLE, READY
///   psum_choose_i           - Specify the psum MUX and DEMUX for DATA, ENABLE, READY
///   compute_i               - Trigger computation
///   data_write_enable_i     - Wether data should be read or written to RAMs. 1 is WR, 0 is RD
///   ext_mem_iact_addr_i     - Sets the address of IACT GLB from top modul
///   ext_mem_iact_data_i     - Input activation data from top modul
///   ext_mem_iact_enable_i   - Enable input activation data transfer from top modul
///   ext_mem_iact_ready_o    - Ready signal for input activation data transfer to top modul
///   ext_mem_wght_data_i     - Weights data from top modul
///   ext_mem_wght_enable_i   - Enable weights data transfer from top modul
///   ext_mem_wght_ready_o    - Ready signal for weight data transfer to top modul
///   ext_mem_psum_addr_i     - Sets the address of PSUM GLB from top modul 
///   ext_mem_psum_data_i     - Partial sum data from top modul
///   ext_mem_psum_enable_i   - Enable partial sum data transfer from top modul
///   ext_mem_psum_ready_o    - Ready signal for partial sum data transfer to top modul
///   ext_mem_psum_data_o     - Partial sum data to top modul
///   ext_mem_psum_enable_o   - Enable partial sum data transfer to top modul
///   ext_mem_psum_ready_i    - Ready signal for partial sum data transfer from top modul
///   bano_cluster_mode_i     - Chooses mode for the batch normalization
///   af_cluster_mode_i       - Chooses mode for the activation function
///   delay_psum_glb_i        - Chooses the needed delay for the psum data
///

module OpenEye_Cluster
#(
  parameter IS_TOPLEVEL                = 1,
  parameter LEFT_CLUSTER               = 0,
  parameter TOP_CLUSTER                = 0,
  parameter BOTTOM_CLUSTER             = 0,
  parameter DATA_IACT_BITWIDTH         = 8,
  parameter DATA_PSUM_BITWIDTH         = 20,
  parameter DATA_WGHT_BITWIDTH         = 8,
  parameter TRANS_BITWIDTH_IACT        = 24,
  parameter TRANS_BITWIDTH_PSUM        = 20,
  parameter TRANS_BITWIDTH_WGHT        = 24,
  parameter NUM_GLB_IACT               = 3,
  parameter NUM_GLB_WGHT               = 3,
  parameter NUM_GLB_PSUM               = 4,
  parameter PE_ROWS                    = 3,
  parameter PE_COLUMNS                 = 4,
  parameter PES                        = PE_ROWS * PE_COLUMNS,
  parameter CLUSTER_ROWS               = 8,
  parameter CLUSTER_COLUMNS             = 2,
  parameter CLUSTERS                   = CLUSTER_COLUMNS * CLUSTER_ROWS,
  parameter IACT_PER_PE                = 16,
  parameter WGHT_PER_PE                = 192,
  parameter PSUM_PER_PE                = 16,
  parameter IACT_MEM_ADDR_WORDS        = 512,
  parameter IACT_MEM_ADDR_BITS         = $clog2(IACT_MEM_ADDR_WORDS),
  parameter PSUM_MEM_ADDR_WORDS        = 384,
  parameter PSUM_MEM_ADDR_BITS         = $clog2(PSUM_MEM_ADDR_WORDS),
  parameter BANO_MODES                 = 2,
  parameter AF_MODES                   = 2
)( 
  input                                          clk_i,
  input                                          rst_ni,
  input [$clog2(NUM_GLB_IACT)*PES-1:0]           iact_choose_i,
  input [NUM_GLB_PSUM-1:0]                       psum_choose_i,
  input [PES-1:0]                                compute_i,
  input                                          data_write_enable_iact_i,
  input                                          data_write_enable_i,

  ///Connections to external Memory
  /////////////////////////////////////////
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   ext_mem_iact_data_i,
  input  [IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0]    ext_mem_iact_addr_i,
  input  [NUM_GLB_IACT-1:0]                       ext_mem_iact_enable_i,
  output [NUM_GLB_IACT-1:0]                       ext_mem_iact_ready_o,

  input  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   ext_mem_wght_data_i,
  input  [NUM_GLB_WGHT-1:0]                       ext_mem_wght_enable_i,
  output [NUM_GLB_WGHT-1:0]                       ext_mem_wght_ready_o,

  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   ext_mem_psum_data_i,
  input  [PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0]    ext_mem_psum_addr_i,
  input  [NUM_GLB_PSUM-1:0]                       ext_mem_psum_enable_i,
  output [NUM_GLB_PSUM-1:0]                       ext_mem_psum_ready_o,

  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   ext_mem_psum_data_o,
  output [NUM_GLB_PSUM-1:0]                       ext_mem_psum_enable_o,
  input  [NUM_GLB_PSUM-1:0]                       ext_mem_psum_ready_i,

  
  ///Router Ports
  /////////////////////////////////////////
  
  ///Weights
  input  [NUM_GLB_WGHT-1:0]                       router_mode_wght_i,
    
  output [NUM_GLB_WGHT-1:0]                       enable_dst_side_wght,
  output [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   data_dst_side_wght,
  input  [NUM_GLB_WGHT-1:0]                       ready_dst_side_wght,
    
  input  [NUM_GLB_WGHT-1:0]                       enable_src_side_wght,
  input  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   data_src_side_wght,
  output [NUM_GLB_WGHT-1:0]                       ready_src_side_wght,
    
  ///Activations
  input  [6*NUM_GLB_IACT-1:0]                     router_mode_iact_i,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_side_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_side_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_side_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_side_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_side_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_side_iact,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_top_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_top_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_top_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_top_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_top_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_top_iact,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_bottom_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_bottom_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_bottom_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_bottom_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_bottom_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_bottom_iact,
    
  ///Psum
  input  [3*NUM_GLB_PSUM-1:0]                     router_mode_psum_i,
    
  output [NUM_GLB_PSUM-1:0]                       enable_dst_top_psum,
  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_dst_top_psum,
  input  [NUM_GLB_PSUM-1:0]                       ready_dst_top_psum,
    
  input  [NUM_GLB_PSUM-1:0]                       enable_src_top_psum,
  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_src_top_psum,
  output [NUM_GLB_PSUM-1:0]                       ready_src_top_psum,
    
  output [NUM_GLB_PSUM-1:0]                       enable_dst_bottom_psum,
  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_dst_bottom_psum,
  input  [NUM_GLB_PSUM-1:0]                       ready_dst_bottom_psum,
    
  input  [NUM_GLB_PSUM-1:0]                       enable_src_bottom_psum,
  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_src_bottom_psum,
  output [NUM_GLB_PSUM-1:0]                       ready_src_bottom_psum,
  
  input                                           enable_stream_i,
  input  [7:0]                                    data_stream_i,

  input  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]    bano_cluster_mode_i,
  input  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]      af_cluster_mode_i,
  input  [3:0]                                    delay_psum_glb_i
);
  ///#######################
  ///Reset synchronization
  ///#######################
  wire rst_n;

  RST_SYNC rst_sync_top (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .rst_no       (rst_n)
  );

  /////////////////////////////////////////
  ///Wires to lead to PEs
  /////////////////////////////////////////
  
  ///WGHT Wire  
  wire [NUM_GLB_WGHT-1:0]                     pe_wght_ready;
  wire [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] pe_wght_data;
  wire [NUM_GLB_WGHT-1:0]                     pe_wght_enable;
  
  ///IACT Wire
  wire [NUM_GLB_IACT-1:0]                     router_cluster_iact_ready;
  wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] router_cluster_iact_data;
  wire [NUM_GLB_IACT-1:0]                     router_cluster_iact_enable;
  
  wire [NUM_GLB_IACT-1:0]                     pe_iact_ready;
  wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] pe_iact_data;
  wire [NUM_GLB_IACT-1:0]                     pe_iact_enable;
  
  ///PSUM Wire  
  wire [NUM_GLB_PSUM-1:0]                     pe_router_psum_ready_in;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] pe_router_psum_data_in;
  wire [NUM_GLB_PSUM-1:0]                     pe_router_psum_enable_in;
  
  wire [NUM_GLB_PSUM-1:0]                     pe_router_psum_ready_out;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] pe_router_psum_data_out;
  wire [NUM_GLB_PSUM-1:0]                     pe_router_psum_enable_out;
  
  
  /////////////////////////////////////////
  ///PEs Cluster
  /////////////////////////////////////////
  
  PE_cluster #(
    .IS_TOPLEVEL(0),
    .TOP_CLUSTER(TOP_CLUSTER),

    .DATA_IACT_BITWIDTH (DATA_IACT_BITWIDTH),
    .DATA_WGHT_BITWIDTH (DATA_WGHT_BITWIDTH),
    .DATA_PSUM_BITWIDTH (DATA_PSUM_BITWIDTH),

    .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
    .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
    .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),

    .NUM_GLB_IACT       (NUM_GLB_IACT),
    
    .PE_ROWS            (PE_ROWS),
    .PE_COLUMNS         (PE_COLUMNS)
  )pe_cluster(
    .clk_i                    (clk_i),
    .rst_ni                   (rst_n),
    .iact_choose_i            (iact_choose_i),
    .psum_choose_i            (psum_choose_i),
    .compute_i                (compute_i),
    

    .pe_iact_data             (pe_iact_data),
    .pe_iact_enable           (pe_iact_enable),
    .pe_iact_ready            (pe_iact_ready),
    
    .pe_wght_data             (pe_wght_data),
    .pe_wght_enable           (pe_wght_enable),
    .pe_wght_ready            (pe_wght_ready),
    
    .pe_psum_ready_i          (ready_dst_top_psum),
    .pe_psum_data_i           (data_src_bottom_psum),
    .pe_psum_enable_i         (enable_src_bottom_psum),

    .pe_psum_ready_o          (ready_src_bottom_psum),
    .pe_psum_data_o           (data_dst_top_psum),
    .pe_psum_enable_o         (enable_dst_top_psum),

    .pe_router_psum_ready_i   (pe_router_psum_ready_in),
    .pe_router_psum_data_i    (pe_router_psum_data_in),
    .pe_router_psum_enable_i  (pe_router_psum_enable_in),

    .pe_router_psum_ready_o   (pe_router_psum_ready_out),
    .pe_router_psum_data_o    (pe_router_psum_data_out),
    .pe_router_psum_enable_o  (pe_router_psum_enable_out),
    .enable_stream_i          (enable_stream_i),
    .data_stream_i            (data_stream_i)

  );


  /////////////////////////////////////////
  ///Wires to lead to GLBs
  /////////////////////////////////////////
  
  ///WGHT Wire
  wire [NUM_GLB_WGHT-1:0]                     glb_cluster_wght_ready;
  wire [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] glb_cluster_wght_data;
  wire [NUM_GLB_WGHT-1:0]                     glb_cluster_wght_enable;
  
  ///IACT Wire
  wire [NUM_GLB_IACT-1:0]                     glb_cluster_iact_ready;
  wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] glb_cluster_iact_data;
  wire [NUM_GLB_IACT-1:0]                     glb_cluster_iact_enable;
  
  ///PSUM Wire  
  wire [NUM_GLB_PSUM-1:0]                     glb_cluster_psum_ready_r;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] glb_cluster_psum_data_r;
  wire [NUM_GLB_PSUM-1:0]                     glb_cluster_psum_enable_r;
  
  wire [NUM_GLB_PSUM-1:0]                     glb_cluster_psum_ready_w;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] glb_cluster_psum_data_w;
  wire [NUM_GLB_PSUM-1:0]                     glb_cluster_psum_enable_w;

  /////////////////////////////////////////
  ///GLB Cluster connect to external memory
  /////////////////////////////////////////
  GLB_cluster #(
    .DATA_IACT_BITWIDTH (TRANS_BITWIDTH_IACT),
    .DATA_PSUM_BITWIDTH (TRANS_BITWIDTH_PSUM),
    .DATA_WGHT_BITWIDTH (TRANS_BITWIDTH_WGHT),
    .NUM_GLB_IACT       (NUM_GLB_IACT),
    .NUM_GLB_WGHT       (NUM_GLB_WGHT),
    .NUM_GLB_PSUM       (NUM_GLB_PSUM),
    .IACT_MEM_ADDR_WORDS(IACT_MEM_ADDR_WORDS),
    .PSUM_MEM_ADDR_WORDS(PSUM_MEM_ADDR_WORDS)
  )glb_cluster (
    .clk_i                       (clk_i),
    .rst_ni                      (rst_n),
    .data_write_enable_iact_i    (data_write_enable_iact_i),
    .data_write_enable_i         (data_write_enable_i),
    
    ///Ext. Memory
    /////////////////////////////////////////
    
    .ext_mem_iact_data_i         (ext_mem_iact_data_i),
    .ext_mem_iact_addr_i         (ext_mem_iact_addr_i),
    .ext_mem_iact_enable_i       (ext_mem_iact_enable_i),
    .ext_mem_iact_ready_o        (ext_mem_iact_ready_o),
    
    .ext_mem_wght_data_i         (ext_mem_wght_data_i),
    .ext_mem_wght_enable_i       (ext_mem_wght_enable_i),
    .ext_mem_wght_ready_o        (ext_mem_wght_ready_o),
    
    .ext_mem_psum_ready_i        (ext_mem_psum_ready_i),
    .ext_mem_psum_data_o         (ext_mem_psum_data_o),
    .ext_mem_psum_enable_o       (ext_mem_psum_enable_o),

    .ext_mem_psum_data_i         (ext_mem_psum_data_i),
    .ext_mem_psum_addr_i         (ext_mem_psum_addr_i),
    .ext_mem_psum_enable_i       (ext_mem_psum_enable_i),
    .ext_mem_psum_ready_o        (ext_mem_psum_ready_o),
  
    ///Router Cluster
    /////////////////////////////////////////
    
    .router_cluster_iact_data_o  (glb_cluster_iact_data),
    .router_cluster_iact_enable_o(glb_cluster_iact_enable),
    .router_cluster_iact_ready_i (glb_cluster_iact_ready),
    
    .router_cluster_wght_data_o  (glb_cluster_wght_data),
    .router_cluster_wght_enable_o(glb_cluster_wght_enable),
    .router_cluster_wght_ready_i (glb_cluster_wght_ready),
    
    .router_cluster_psum_data_o  (glb_cluster_psum_data_r),
    .router_cluster_psum_enable_o(glb_cluster_psum_enable_r),
    .router_cluster_psum_ready_i (glb_cluster_psum_ready_r),
    
    .router_cluster_psum_data_i  (glb_cluster_psum_data_w),
    .router_cluster_psum_enable_i(glb_cluster_psum_enable_w),
    .router_cluster_psum_ready_o (glb_cluster_psum_ready_w)
  );


  /////////////////////////////////////////
  ///Post Result Collecting from PEs
  /////////////////////////////////////////

  /////////////////////////////////////////
  ///Wires to lead to Batch Normalization
  /////////////////////////////////////////

  wire [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]  bano_cluster_mode_in;

  wire [NUM_GLB_PSUM-1:0]                     bano_cluster_ready_in;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] bano_cluster_data_in;
  wire [NUM_GLB_PSUM-1:0]                     bano_cluster_enable_in;

  wire [NUM_GLB_PSUM-1:0]                     bano_cluster_ready_out;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] bano_cluster_data_out;
  wire [NUM_GLB_PSUM-1:0]                     bano_cluster_enable_out;

  /////////////////////////////////////////
  ///Batch Normalization Cluster
  /////////////////////////////////////////

  genvar r, b;
  generate for(r=0; r<NUM_GLB_PSUM; r=r+1)begin : bano_cluster_gen
    bano_cluster #(
      .DATA_BITWIDTH(TRANS_BITWIDTH_PSUM)
    )bano_cluster(
      .clk_i       (clk_i),
      .rst_ni      (rst_n),

      .ready_o     (bano_cluster_ready_out[r]),
      .data_i      (bano_cluster_data_in[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_i    (bano_cluster_enable_in[r]),

      .ready_i     (bano_cluster_ready_in[r]),
      .data_o      (bano_cluster_data_out[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_o    (bano_cluster_enable_out[r])
    );
  end
  endgenerate

  /////////////////////////////////////////
  ///Wires to lead to Pooling
  /////////////////////////////////////////

  wire [NUM_GLB_PSUM-1:0]                       delay_cluster_ready_in;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   delay_cluster_data_in;
  wire [NUM_GLB_PSUM-1:0]                       delay_cluster_enable_in;

  wire [NUM_GLB_PSUM-1:0]                       delay_cluster_ready_out;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   delay_cluster_data_out;
  wire [NUM_GLB_PSUM-1:0]                       delay_cluster_enable_out;

  /////////////////////////////////////////
  ///Pooling Cluster
  /////////////////////////////////////////

  generate for(r=0; r<NUM_GLB_PSUM; r=r+1)begin : delay_cluster_gen
    delay_cluster #(
      .DATA_BITWIDTH(TRANS_BITWIDTH_PSUM)
    )delay_cluster(
      .clk_i       (clk_i),
      .rst_ni      (rst_n),

      .ready_o     (delay_cluster_ready_out[r]),
      .data_i      (delay_cluster_data_in[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_i    (delay_cluster_enable_in[r]),

      .ready_i     (delay_cluster_ready_in[r]),
      .data_o      (delay_cluster_data_out[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_o    (delay_cluster_enable_out[r]),

      .delay_psum_glb_i(delay_psum_glb_i)
    );
  end
  endgenerate
  /////////////////////////////////////////
  ///Wires to lead to Activation Functions
  /////////////////////////////////////////

  wire [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]    af_cluster_mode_in;
  assign af_cluster_mode_in = af_cluster_mode_i;

  wire [NUM_GLB_PSUM-1:0]                     af_cluster_ready_in;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] af_cluster_data_in;
  wire [NUM_GLB_PSUM-1:0]                     af_cluster_enable_in;

  wire [NUM_GLB_PSUM-1:0]                     af_cluster_ready_out;
  wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] af_cluster_data_out;
  wire [NUM_GLB_PSUM-1:0]                     af_cluster_enable_out;

  /////////////////////////////////////////
  ///AF Cluster
  /////////////////////////////////////////

  generate for(r=0; r<NUM_GLB_PSUM; r=r+1)begin : af_cluster_gen
    af_cluster #(
      .DATA_BITWIDTH(TRANS_BITWIDTH_PSUM),

      .MODES(AF_MODES)
    )af_cluster(
      .clk_i            (clk_i),
      .rst_ni           (rst_n),

      .mode_i           (af_cluster_mode_in[$clog2(AF_MODES)*(r+1)-1:$clog2(AF_MODES)*r]),

      .ready_o          (af_cluster_ready_out[r]),
      .data_i           (af_cluster_data_in[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_i         (af_cluster_enable_in[r]),

      .ready_i          (af_cluster_ready_in[r]),
      .data_o           (af_cluster_data_out[TRANS_BITWIDTH_PSUM*(r+1)-1:TRANS_BITWIDTH_PSUM*r]),
      .enable_o         (af_cluster_enable_out[r])
    );
  end
  endgenerate

  /////////////////////////////////////////
  ///Assigments for post-collected results
  /////////////////////////////////////////
  for(r=0; r<NUM_GLB_PSUM; r=r+1)begin : post_result_assig_gen_cluster

    assign af_cluster_ready_in[r]       = delay_cluster_ready_out[r];
    assign delay_cluster_enable_in[r] = af_cluster_enable_out[r];

    assign delay_cluster_ready_in[r]  = bano_cluster_ready_out[r];
    assign bano_cluster_enable_in[r]    = delay_cluster_enable_out[r];

    assign bano_cluster_ready_in[r]     = glb_cluster_psum_ready_w[r];
    assign glb_cluster_psum_enable_w[r] = bano_cluster_enable_out[r];

    for(b=0; b<TRANS_BITWIDTH_PSUM; b=b+1)begin : post_result_assig_gen_bits
      assign delay_cluster_data_in[TRANS_BITWIDTH_PSUM*r+b] = af_cluster_data_out[TRANS_BITWIDTH_PSUM*r+b];
      assign bano_cluster_data_in[TRANS_BITWIDTH_PSUM*r+b]    = delay_cluster_data_out[TRANS_BITWIDTH_PSUM*r+b];
      assign glb_cluster_psum_data_w[TRANS_BITWIDTH_PSUM*r+b] = bano_cluster_data_out[TRANS_BITWIDTH_PSUM*r+b];
    end
  end

  /////////////////////////////////////////
  ///Router Clusters
  /////////////////////////////////////////
        
  genvar i;
  generate for (i=0; i<NUM_GLB_IACT; i = i+1) begin : router_iact_gen
    router_iact #(
      .LEFT_CLUSTER(LEFT_CLUSTER),
      .DATA_WIDTH(TRANS_BITWIDTH_IACT)
    )router_iact(
      .router_mode_i    (router_mode_iact_i[6*(i+1)-1:6*i]),
      ///SRC Port 0
      ///Conncet to: GLB
      .ready_src_port_0 (glb_cluster_iact_ready[i]),
      .data_src_port_0  (glb_cluster_iact_data[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_src_port_0(glb_cluster_iact_enable[i]),
      ///SRC Port 1
      ///Conncet to: Router/Port
      .ready_src_port_1 (ready_src_top_iact[i]),
      .data_src_port_1  (data_src_top_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_src_port_1(enable_src_top_iact[i]),
      ///SRC Port 2
      ///Conncet to: Router/Port
      .ready_src_port_2 (ready_src_side_iact[i]),
      .data_src_port_2  (data_src_side_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_src_port_2(enable_src_side_iact[i]),
      ///SRC Port 3
      ///Conncet to: Router/Port
      .ready_src_port_3 (ready_src_bottom_iact[i]),
      .data_src_port_3  (data_src_bottom_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_src_port_3(enable_src_bottom_iact[i]),
      ///DST Port 0
      ///Conncet to: PEs
      .ready_dst_port_0 (pe_iact_ready[i]),
      .data_dst_port_0  (pe_iact_data[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_dst_port_0(pe_iact_enable[i]),
      ///DST Port 1
      ///Conncet to: Router/Port
      .ready_dst_port_1 (ready_dst_top_iact[i]),
      .data_dst_port_1  (data_dst_top_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_dst_port_1(enable_dst_top_iact[i]),
      ///DST Port 2
      ///Conncet to: Router/Port
      .ready_dst_port_2 (ready_dst_side_iact[i]),
      .data_dst_port_2  (data_dst_side_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_dst_port_2(enable_dst_side_iact[i]),
      ///DST Port 3
      ///Conncet to: Router/Port
      .ready_dst_port_3 (ready_dst_bottom_iact[i]),
      .data_dst_port_3  (data_dst_bottom_iact[TRANS_BITWIDTH_IACT*(i+1)-1:TRANS_BITWIDTH_IACT*i]),
      .enable_dst_port_3(enable_dst_bottom_iact[i])
    );
  end
  endgenerate

  genvar j;
  generate for(j=0; j<NUM_GLB_PSUM; j=j+1) begin : router_psum_gen
    router_psum #(
      .DATA_WIDTH(TRANS_BITWIDTH_PSUM)
    )router_psum(
      .router_mode_i    (router_mode_psum_i[3*(j+1)-1:3*j]),
      ///SRC Port 0
      ///Conncet to: GLB
      .ready_src_port_0 (glb_cluster_psum_ready_r[j]),
      .data_src_port_0  (glb_cluster_psum_data_r[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_src_port_0(glb_cluster_psum_enable_r[j]),
      ///SRC Port 1
      ///Conncet to: PEs
      .ready_src_port_1 (pe_router_psum_ready_in[j]),
      .data_src_port_1  (pe_router_psum_data_out[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_src_port_1(pe_router_psum_enable_out[j]),
      ///SRC Port 2
      ///Conncet to: Router/Port
      .ready_src_port_2 (ready_src_top_psum[j]),
      .data_src_port_2  (data_src_top_psum[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_src_port_2(enable_src_top_psum[j]),
      ///DST Port 0
      ///Conncet to: AF Cluster
      .ready_dst_port_0 (af_cluster_ready_out[j]),
      .data_dst_port_0  (af_cluster_data_in[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_dst_port_0(af_cluster_enable_in[j]),
      ///DST Port 1
      ///Conncet to: PEs
      .ready_dst_port_1 (pe_router_psum_ready_out[j]),
      .data_dst_port_1  (pe_router_psum_data_in[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_dst_port_1(pe_router_psum_enable_in[j]),
      ///DST Port 2
      ///Conncet to: Router/Port
      .ready_dst_port_2 (ready_dst_bottom_psum[j]),
      .data_dst_port_2  (data_dst_bottom_psum[TRANS_BITWIDTH_PSUM*(j+1)-1:TRANS_BITWIDTH_PSUM*j]),
      .enable_dst_port_2(enable_dst_bottom_psum[j])
    );
  end
  endgenerate

  genvar k;
  generate for(k=0; k<NUM_GLB_WGHT; k=k+1) begin : router_wght_gen

    router_wght #(
        .DATA_WIDTH(TRANS_BITWIDTH_WGHT)
    )router_wght(
        .router_mode_i    (router_mode_wght_i[k]),
        ///SRC Port 0
        ///Conncet to: GLB
        .ready_src_port_0 (glb_cluster_wght_ready[k]),
        .data_src_port_0  (glb_cluster_wght_data[TRANS_BITWIDTH_WGHT*(k+1)-1:TRANS_BITWIDTH_WGHT*k]),
        .enable_src_port_0(glb_cluster_wght_enable[k]),
        ///SRC Port 1
        ///Conncet to: Router/Port
        .ready_src_port_1 (ready_src_side_wght[k]),
        .data_src_port_1  (data_src_side_wght[TRANS_BITWIDTH_WGHT*(k+1)-1:TRANS_BITWIDTH_WGHT*k]),
        .enable_src_port_1(enable_src_side_wght[k]),
        ///DST Port 0
        ///Conncet to: PEs
        .ready_dst_port_0 (pe_wght_ready[k]),
        .data_dst_port_0  (pe_wght_data[TRANS_BITWIDTH_WGHT*(k+1)-1:TRANS_BITWIDTH_WGHT*k]),
        .enable_dst_port_0(pe_wght_enable[k]),
        ///DST Port 1
        ///Conncet to: Router/Port
        .ready_dst_port_1 (ready_dst_side_wght[k]),
        .data_dst_port_1  (data_dst_side_wght[TRANS_BITWIDTH_WGHT*(k+1)-1:TRANS_BITWIDTH_WGHT*k]),
        .enable_dst_port_1(enable_dst_side_wght[k])
      );
  end
  endgenerate
endmodule
