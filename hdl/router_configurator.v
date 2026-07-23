// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_configurator
///
/// Overview:
/// This module encapsulates the dynamic router-mode configuration logic that was
/// previously implemented as "Process 7" inside OpenEye_FPGA. It updates the
/// routing vectors (iact / wght / psum) sent to OpenEye_Parallel while computation
/// is in progress. This is separate from the static GET_ROUTER_CONFIG load in the
/// main FSM: it handles mid-computation iact-routing rotation and psum-routing
/// updates that must track the evolving iact delivery schedule.
///
/// Local registers:
///   router_mode_iact_storage         - saved copy of the iact router vector;
///                                      restored at the start of each weight-reuse
///                                      iteration so the iact routing pattern
///                                      repeats correctly.
///   storage_cycles_router            - cycle counter for shifting psum router
///                                      output-enable bits through the cluster rows.
///   first_cycle                      - flag: 1 on the very first compute iteration;
///                                      prevents premature router updates.
///   psum_choose_i_reg                - per-cluster source-select for psum routing;
///                                      forwarded to OpenEye_Parallel's psum_choose_i.
///   iact_channels_counter_psum_router - local copy of iact_channels_counter used
///                                      inside this process to avoid combinational
///                                      dependencies across processes.
///
/// Key behaviour:
///   WAIT_FOR_RESULTS / RECEIVE_PSUMS_TO_IACT:
///     On each single_iteration3 pulse (new iact delivery):
///       - If iact_cycle_count wraps (== needed_wght_cycles - 1):
///           restores router_mode_iact from router_mode_iact_storage (reset routing).
///       - Otherwise: rotates router_mode_iact left by the cluster-column stride
///           (CLUSTER_COLUMNS * ROUTER_MODES_IACT * NUM_GLB_IACT bits) so the
///           active iact source moves along the cluster columns each cycle.
///     On single_iteration pulse: updates psum router bits to steer partial sums
///       from the correct cluster row into the psum buffer; shifts the output-row
///       enable (bit [2]) down the CLUSTER_ROWS chain via storage_cycles_router.
///   reset_cycle: clears iact routing and counters.
///
/// Parameters:
///   CLUSTER_COLUMNS        - Number of cluster columns
///   CLUSTER_ROWS           - Number of cluster rows
///   NUM_GLB_IACT           - Number of activation global buffers
///   NUM_GLB_WGHT           - Number of weight global buffers
///   NUM_GLB_PSUM           - Number of partial-sum global buffers
///   CLUSTERS               - Total number of clusters
///   DMA_BITWIDTH           - DMA data width
///   ROUTER_MODES_IACT      - Router mode count for activation traffic
///   ROUTER_MODES_WGHT      - Router mode count for weight traffic
///   ROUTER_MODES_PSUM      - Router mode count for partial-sum traffic
///   FSM_CEIL_IACT_RTR_CCLS - Ceil of iact router config cycles
///   FSM_CEIL_WGHT_RTR_CCLS - Ceil of wght router config cycles
///   FSM_CEIL_PSUM_RTR_CCLS - Ceil of psum router config cycles
///   fsm_psum_rTR_CCLS_C    - DMA bit width aligned to ROUTER_MODES_PSUM
///
/// Ports:
///   clk_i                          - System clock input
///   rst_n                          - Active-low reset input
///   compute_reg_i                  - One-cycle pulse triggering a computation batch
///   fully_connected_layer_i        - When 1: layer is a fully-connected (FC) layer
///   needed_y_cls_reg_i             - Number of active cluster rows for this layer
///   fsm_current_state_i            - Current state of the main FSM
///   enable_dma_i_reg_i             - Registered DMA input valid flag
///   fsm_cycle_i                    - General-purpose per-state cycle counter
///   data_dma_i_reg_i               - Registered DMA input data word
///   single_iteration3_i            - Single-cycle pulse on delivery start
///   iact_channels_counter_i        - Counts which channel batch is being processed
///   iact_channel_max_cycles_i      - Total number of channel batches per layer pass
///   iact_router_counter_i          - Counts cluster-row sweeps within one iact batch
///   needed_psum_storage_cycles_i   - PSUM accumulation passes required
///   reset_cycle_i                  - Pulse that resets all iteration counters to 0
///   router_mode_iact_o             - Iact router configuration output
///   router_mode_wght_o             - Weight router configuration output
///   router_mode_psum_o             - Psum router configuration output
///   router_mode_iact_storage_o     - Saved iact router vector output
///   storage_cycles_router_o        - Psum router shift cycle counter output
///   first_cycle_o                  - First compute iteration flag output
///   psum_choose_i_reg_o            - Psum routing source-select output
///   iact_channels_counter_psum_router_o - Local channel counter copy output
///
module router_configurator #(
    parameter CLUSTER_COLUMNS = 2,
    parameter CLUSTER_ROWS    = 2,
    parameter NUM_GLB_IACT    = 3,
    parameter NUM_GLB_WGHT    = 3,
    parameter NUM_GLB_PSUM    = 4,
    parameter CLUSTERS        = CLUSTER_COLUMNS * CLUSTER_ROWS,
    parameter DMA_BITWIDTH    = 64,

    parameter ROUTER_MODES_IACT = 6,
    parameter ROUTER_MODES_WGHT = 1,
    parameter ROUTER_MODES_PSUM = 3,

    parameter integer FSM_CEIL_IACT_RTR_CCLS = 1,
    parameter integer FSM_CEIL_WGHT_RTR_CCLS = 1,
    parameter integer FSM_CEIL_PSUM_RTR_CCLS = 1,
    parameter integer fsm_psum_rTR_CCLS_C    = DMA_BITWIDTH - (DMA_BITWIDTH % ROUTER_MODES_PSUM)
) (
    // Clock and Reset
    input clk_i,
    input rst_n,

    // Control inputs from main FSM / Process 2
    input compute_reg_i,
    input fully_connected_layer_i,
    input [$clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg_i,
    input [3:0] fsm_current_state_i,
    input enable_dma_i_reg_i,
    input [32-1:0] fsm_cycle_i,
    input [DMA_BITWIDTH-1:0] data_dma_i_reg_i,
    input single_iteration3_i,
    input [$clog2(16+1)-1:0] iact_channels_counter_i,
    input [7:0] iact_channel_max_cycles_i,
    input [$clog2(CLUSTER_ROWS)-1:0] iact_router_counter_i,
    input [7:0] needed_psum_storage_cycles_i,
    input reset_cycle_i,

    // Router configuration outputs
    output reg [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_o,
    output reg [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght_o,
    output reg [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum_o,

    // Local register outputs
    output reg [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_storage_o,
    output reg [7:0] storage_cycles_router_o,
    output reg first_cycle_o,
    output reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_choose_i_reg_o,
    output reg [7:0] iact_channels_counter_psum_router_o
);

  // Loop variables
  integer cc, cr, g;

  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      router_mode_iact_o                  <= 0;
      router_mode_iact_storage_o          <= 0;
      router_mode_wght_o                  <= 0;
      router_mode_psum_o                  <= 0;
      storage_cycles_router_o             <= 0;
      first_cycle_o                       <= 1;
      psum_choose_i_reg_o                 <= 0;
      iact_channels_counter_psum_router_o <= 0;
    end else begin
      if (compute_reg_i) begin
        if (fully_connected_layer_i) begin
          psum_choose_i_reg_o <= {CLUSTER_COLUMNS{{NUM_GLB_PSUM{1'b1}}, {((CLUSTER_ROWS - 1) * NUM_GLB_PSUM){1'b0}}}};
        end else begin
          if (needed_y_cls_reg_i == 1) begin
            psum_choose_i_reg_o <= (2 ** (CLUSTER_ROWS * CLUSTER_COLUMNS * NUM_GLB_PSUM) - 1);
          end else begin
            if (needed_y_cls_reg_i == 2) begin
              psum_choose_i_reg_o <= {CLUSTER_ROWS{{NUM_GLB_PSUM{1'b1}}, {NUM_GLB_PSUM{1'b0}}}};
            end else begin
              if (CLUSTER_ROWS == 8) begin
                psum_choose_i_reg_o <= {((CLUSTER_ROWS+1)/2){{NUM_GLB_PSUM{1'b1}},{NUM_GLB_PSUM{3'd0}}}};
              end else begin
                psum_choose_i_reg_o <= {(4){{NUM_GLB_PSUM{1'b1}},{NUM_GLB_PSUM{2'd0}}}};
              end
            end
          end
        end
      end
      if (CLUSTERS!= 1) begin
        if (fsm_current_state_i == 4'd2) begin // GET_ROUTER_CONFIG
          storage_cycles_router_o  <= 0;
          first_cycle_o            <= 1;
          iact_channels_counter_psum_router_o <= 0;
          if (enable_dma_i_reg_i) begin
            if (fsm_cycle_i < FSM_CEIL_IACT_RTR_CCLS) begin
              for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                  for (g = 0; g < NUM_GLB_IACT; g = g + 1) begin
                    if(((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)>=(fsm_cycle_i    *(DMA_BITWIDTH/ROUTER_MODES_IACT)))
                      &((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)< ((fsm_cycle_i+1)*(DMA_BITWIDTH/ROUTER_MODES_IACT))))begin
                      router_mode_iact_o[cc * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                                          cr * NUM_GLB_IACT * ROUTER_MODES_IACT +
                                          g * ROUTER_MODES_IACT +:ROUTER_MODES_IACT] <=
                      data_dma_i_reg_i[(cc*NUM_GLB_IACT+cr*CLUSTER_COLUMNS*NUM_GLB_IACT+g-fsm_cycle_i*(DMA_BITWIDTH/ROUTER_MODES_IACT))
                      *ROUTER_MODES_IACT+:ROUTER_MODES_IACT];
                    end
                  end
                end
              end
            end else begin
              if (fsm_cycle_i < FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS) begin
                for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                  for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                    for (g = 0; g < NUM_GLB_WGHT; g = g + 1) begin
                      if(((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)>=((fsm_cycle_i-FSM_CEIL_IACT_RTR_CCLS)  *(DMA_BITWIDTH/ROUTER_MODES_WGHT)))
                        &((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)< ((fsm_cycle_i+1-FSM_CEIL_IACT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_WGHT))))begin
                        router_mode_wght_o[cc * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                            cr * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                            g * ROUTER_MODES_WGHT+: ROUTER_MODES_WGHT] <=
                        data_dma_i_reg_i[(cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g-(fsm_cycle_i-FSM_CEIL_IACT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_WGHT))+:ROUTER_MODES_WGHT];
                      end
                    end
                  end
                end
              end else begin
                for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                  for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                    for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                      if(((cc*CLUSTER_ROWS*NUM_GLB_PSUM+cr*NUM_GLB_PSUM+g)>=((fsm_cycle_i-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)  *(DMA_BITWIDTH/ROUTER_MODES_PSUM)))
                        &((cc*CLUSTER_ROWS*NUM_GLB_PSUM+cr*NUM_GLB_PSUM+g)< ((fsm_cycle_i+1-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_PSUM))))begin
                        router_mode_psum_o[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                                            cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                                            g * ROUTER_MODES_PSUM +:ROUTER_MODES_PSUM] <=
                        data_dma_i_reg_i[(cc*CLUSTER_ROWS*NUM_GLB_PSUM*ROUTER_MODES_PSUM+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM-(fsm_cycle_i-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*fsm_psum_rTR_CCLS_C)+:ROUTER_MODES_PSUM];
                      end
                    end
                  end
                end
              end
            end
          end
          router_mode_iact_storage_o <= router_mode_iact_o;
        end else begin
          if (fsm_current_state_i == 4'd11 | fsm_current_state_i == 4'd12) begin // WAIT_FOR_RESULTS | RECEIVE_PSUMS_TO_IACT
            if (single_iteration3_i) begin
              if (iact_channels_counter_i == iact_channel_max_cycles_i -1) begin
                if (iact_router_counter_i == needed_y_cls_reg_i - 1) begin
                  router_mode_iact_o <= router_mode_iact_storage_o;
                end else begin
                  for (cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (g=0; g<NUM_GLB_IACT; g=g+1) begin
                      router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+3] <= 0;
                      router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+4] <= 1;
                      router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+5] <= 1;
                    end
                  end
                  for (cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                    for (cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                      for (g=0; g<NUM_GLB_IACT; g=g+1) begin
                        // If router is not on top of source already
                        if (!((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT] == 1) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] == 1) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+2] == 0) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 0) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1))) begin
                          // If router is source
                          if ((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                                // If router above is already destination
                                if ((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                    (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                                  router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                                end
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                          end else begin
                            if ((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                                (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                              router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                              router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 0;
                              router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
                            end else begin
                              if ((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                  (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                                router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                              end else begin
                                if ((router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 1) &
                                    (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                    (router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin

                                    router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                                    router_mode_iact_o[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
                                end
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
              //PSUM Router
              first_cycle_o    <= 0;
              if (first_cycle_o == 0) begin
                if ((needed_y_cls_reg_i >= 2)) begin
                  iact_channels_counter_psum_router_o <= iact_channels_counter_psum_router_o + 1;
                  if ((iact_channels_counter_psum_router_o == iact_channel_max_cycles_i - 1)) begin
                    iact_channels_counter_psum_router_o <= 0;
                    for (cr = 1; cr < CLUSTER_ROWS; cr = cr + 1) begin
                      for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                          router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                          router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                        end
                      end
                    end
                    if (storage_cycles_router_o != (needed_psum_storage_cycles_i - 1)) begin
                      storage_cycles_router_o <= storage_cycles_router_o + 1;
                      for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                          router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 0;
                        end
                      end
                      for (cr = 1; cr < CLUSTER_ROWS; cr = cr + 1) begin
                        for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                          for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                            router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                            router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                          end
                        end
                      end
                    end else begin
                      for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                          router_mode_psum_o[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 1;
                        end
                      end
                      storage_cycles_router_o <= 0;
                    end
                  end
                  if (storage_cycles_router_o == needed_psum_storage_cycles_i - 1) begin
                    storage_cycles_router_o <= 0;
                  end else begin
                    storage_cycles_router_o <= storage_cycles_router_o + 1;
                  end
                end
              end
            end
          end
        end
      end
      if (reset_cycle_i) begin
        router_mode_iact_o              <= 0;
        router_mode_iact_storage_o          <= 0;
        storage_cycles_router_o             <= 0;
        first_cycle_o                       <= 1;
        psum_choose_i_reg_o                 <= 0;
        iact_channels_counter_psum_router_o <= 0;
      end
    end
  end

endmodule