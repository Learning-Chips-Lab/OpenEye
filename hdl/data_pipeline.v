// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: data_pipeline
///
/// The data_pipeline module is a critical data movement and formatting component of the OpenEye 
/// neural network accelerator. It orchestrates efficient data transfer between external interfaces
/// and internal scratchpad memories while handling data format transformations and address management.
///
/// Architecture Overview:
/// - Dual Scratchpad System:
///   * First Scratchpad (Address Phase): Manages addressing and data organization
///   * Second Scratchpad (Data Phase): Handles actual data storage and processing
///
/// - Pipeline Stages:
///   1. Address Loading:
///      * Processes incoming data for address information
///      * Performs zero detection for sparse data optimization
///      * Manages address space allocation and bounds checking
///
///   2. Data Loading:
///      * Handles data formatting and alignment
///      * Manages cycle-accurate data transfers
///      * Controls data width adaptation between interfaces
///
/// Key Features:
/// - Variable Data Width Support: Handles different input and output data widths
/// - Cycle-Accurate Control: Precise timing for data movement operations
/// - Zero Detection: Optimizes processing for sparse neural networks
/// - Configurable Memory Depths: Adjustable scratchpad sizes for different requirements
/// - Address Space Protection: Prevents buffer overflows and invalid accesses
///
/// Performance Optimizations:
/// - Efficient State Transitions: Minimizes latency between phases
/// - Early Termination: Detects completion conditions to reduce cycles
/// - Parallel Processing: Concurrent address and data handling
/// - Flexible Data Formatting: Supports various data width configurations
///
/// FSM States:
///   LOADING_ADDR (0):
///     - Address Phase Operation:
///       * Processes incoming address information
///       * Manages address space allocation in first SPAD
///       * Performs bounds checking against FIRST_SPAD_ADDR limit
///       * Tracks valid word count for subsequent operations
///
///     - Control Logic:
///       * Handles data_mode selection for operation mode
///       * Implements zero-detection for early termination
///       * Manages cycle counting for multi-cycle operations
///       * Controls state transitions based on completion conditions
///
///     - Completion Conditions:
///       * Address space limit reached (FIRST_SPAD_ADDR - 1)
///       * Maximum address limit hit (first_spad_max_w)
///       * Zero detection in remaining data
///       * Complete cycle count reached
///
///   LOADING_DATA (1):
///     - Data Phase Operation:
///       * Transfers actual data content to second SPAD
///       * Manages data width adaptation between interfaces
///       * Controls data alignment and formatting
///       * Maintains word count tracking
///
///     - Transfer Control:
///       * Implements cycle-accurate data movement
///       * Manages data storage shifting operations
///       * Controls write enable signals for second SPAD
///       * Handles boundary conditions and overflow protection
///
///     - State Management:
///       * Tracks completion of data phase
///       * Controls transition back to address phase
///       * Manages reset conditions
///       * Handles enable/disable functionality
///
/// Parameters:
/// Data Width Configuration:
///   CALC_DATA_WIDTH            - Internal calculation and counter width (default: 32)
///                               Used for precise arithmetic and cycle counting
///   DATA_WIDTH                 - External interface data bus width
///                               Defines the width of input/output data paths
///
/// First Scratchpad Configuration:
///   FIRST_SPAD_ADDR           - Depth of first scratchpad memory
///                               Controls address storage capacity
///   FIRST_SPAD_DATA           - Data width per entry in first scratchpad
///                               Typically smaller for address storage efficiency
///   FIRST_SPAD_ADDR_BITWIDTH  - Address bits for first scratchpad
///                               Auto-calculated as log2(FIRST_SPAD_ADDR)
///   FIRST_SPAD_DATA_CYCLE     - Number of cycles per first SPAD operation
///                               Derived from DATA_WIDTH/FIRST_SPAD_DATA
///
/// Second Scratchpad Configuration:
///   SECOND_SPAD_ADDR          - Depth of second scratchpad memory
///                               Controls data storage capacity
///   SECOND_SPAD_DATA          - Data width per entry in second scratchpad
///                               Typically wider for efficient data storage
///   SECOND_SPAD_ADDR_BITWIDTH - Address bits for second scratchpad
///                               Auto-calculated as log2(SECOND_SPAD_ADDR)
///   SECOND_SPAD_DATA_CYCLE    - Number of cycles per second SPAD operation
///                               Derived from DATA_WIDTH/SECOND_SPAD_DATA
///
/// Ports:
/// System Interface:
///   clk_i                     - System clock input
///                               Synchronizes all pipeline operations
///   rst_ni                    - Asynchronous reset (active low)
///                               Initializes pipeline state and memory interfaces
///   compute_i                 - Computation/reset trigger
///                               Initiates new processing sequence or resets current operation
///   data_mode                 - Operation mode selection
///                               Controls pipeline behavior and data handling
///   enable_i                  - Processing enable
///                               Gates pipeline operation and data movement
///
/// Data Interface:
///   data_i[DATA_WIDTH-1:0]    - Main input data bus
///                               Carries both address and data information
///
/// First Scratchpad Interface (Address Phase):
///   first_spad_words_o        - Number of valid words processed
///                               Tracks address phase completion
///   first_spad_max_i          - Maximum allowable address
///                               Prevents buffer overflow
///   first_spad_addr_o         - Current address output
///                               Controls scratchpad write location
///   first_spad_data_o         - Address data output
///                               Data to be written to scratchpad
///   first_spad_en_o           - Write enable signal
///                               Controls scratchpad write timing
///
/// Second Scratchpad Interface (Data Phase):
///   second_spad_words_o       - Number of valid data words
///                               Tracks data phase completion
///   second_spad_addr_o        - Current data address
///                               Controls data write location
///   second_spad_data_o        - Main data output
///                               Processed data for storage
///   second_spad_en_o          - Data write enable
///                               Controls data write timing
///
/// Implementation Notes:
/// State Machine Design:
///   - Two-phase FSM implements separate address and data loading phases
///   - State transitions optimized for minimal latency
///   - Comprehensive state reset and recovery mechanisms
///   - Clean separation between control and data paths
///
/// Memory Management:
///   - Dual scratchpad architecture for address and data storage
///   - Independent address spaces with overflow protection
///   - Configurable memory depths and widths
///   - Efficient data alignment and formatting
///
/// Performance Features:
///   - Zero-detection logic for early termination
///   - Cycle-accurate operation with precise timing
///   - Parallel processing of address and data phases
///   - Optimized data path for minimal latency
///
/// Data Flow Control:
///   - Flexible data width adaptation
///   - Support for variable operation modes
///   - Built-in data storage management
///   - Efficient data shifting and alignment
///
/// Reliability Features:
///   - Comprehensive reset handling
///   - Address space overflow protection
///   - Data integrity preservation
///   - Clean state transitions
///
/// Integration Considerations:
///   - Compatible with OpenEye PE array architecture
///   - Supports sparse neural network operations
///   - Configurable for different memory hierarchies
///   - Adaptable to various data formats
///
/// Testability Features:
///   - Observable state transitions
///   - Controllable operation modes
///   - Monitored data paths
///   - Verifiable memory accesses
///

module data_pipeline #(
    parameter CALC_DATA_WIDTH           = 32,
    parameter DATA_WIDTH                = 24,
    parameter FIRST_SPAD_ADDR           = 16,
    parameter FIRST_SPAD_DATA           = 8,
    parameter SECOND_SPAD_ADDR          = 96,
    parameter SECOND_SPAD_DATA          = 24,
    parameter FIRST_SPAD_ADDR_BITWIDTH  = $clog2(FIRST_SPAD_ADDR),
    parameter SECOND_SPAD_ADDR_BITWIDTH = $clog2(SECOND_SPAD_ADDR),
    parameter FIRST_SPAD_DATA_CYCLE     = DATA_WIDTH / FIRST_SPAD_DATA,
    parameter SECOND_SPAD_DATA_CYCLE    = DATA_WIDTH / SECOND_SPAD_DATA
) (
    input clk_i,
    input rst_ni,
    input compute_i,
    input data_mode,

    input [DATA_WIDTH-1 : 0] data_i,
    input                    enable_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [   $clog2(FIRST_SPAD_ADDR)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output     [         FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                  first_spad_en_o,

    output reg [SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output     [         SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                   second_spad_en_o
);

  localparam LOADING_ADDR = 1'b0;
  localparam LOADING_DATA = 1'b1;

  reg                          current_state;  // Current state of the FSM
  reg                          fsm_state;  // FSM state register
  reg  [     DATA_WIDTH-1 : 0] data_storage;  // Temporary storage for data
  reg  [CALC_DATA_WIDTH-1 : 0] address_temp;  // Temporary address storage
  reg  [CALC_DATA_WIDTH-1 : 0] cycle_counter;  // Cycle counter for data loading
  wire [  CALC_DATA_WIDTH-1:0] first_spad_max_w;

  assign first_spad_max_w = {{CALC_DATA_WIDTH - $clog2(FIRST_SPAD_ADDR) {1'b0}}, first_spad_max_i};
  genvar i;
  for (i = 0; i < FIRST_SPAD_DATA; i = i + 1) begin
    assign first_spad_data_o[i] = data_storage[i];  // Assign data to the first SPAD output
  end

  for (i = 0; i < SECOND_SPAD_DATA; i = i + 1) begin
    assign second_spad_data_o[i] = data_storage[i];  // Assign data to the second SPAD output
  end

  always @(*) begin : fsm_state_comb
    fsm_state = current_state;  // Combinational logic to update FSM state
  end


  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      current_state       <= LOADING_ADDR;
      first_spad_words_o  <= 0;
      second_spad_words_o <= 0;
      first_spad_addr_o   <= 0;
      first_spad_en_o     <= 0;
      second_spad_addr_o  <= 0;
      second_spad_en_o    <= 0;
      data_storage        <= 0;
      address_temp        <= 0;
      cycle_counter       <= 0;
    end else begin
      case (fsm_state)
        LOADING_ADDR: begin  // First state, start loading the first SPAD
          second_spad_en_o   <= 0;
          second_spad_addr_o <= 0;
          if ((enable_i == 1) & (!data_mode)) begin
            second_spad_words_o <= 0;
            first_spad_en_o     <= 1;
            first_spad_addr_o   <= address_temp[FIRST_SPAD_ADDR_BITWIDTH-1:0];
            address_temp        <= address_temp + 1;
            cycle_counter       <= cycle_counter + 1;
            data_storage        <= data_storage >> FIRST_SPAD_DATA;
            if (cycle_counter == 0) begin
              data_storage <= data_i;
              if ((data_i == 0) & (first_spad_data_o != 0)) begin
                first_spad_words_o <= first_spad_addr_o + 1;
                current_state      <= LOADING_DATA;
                address_temp       <= 0;
                cycle_counter      <= 0;
              end
            end else begin
              if (((data_storage >> FIRST_SPAD_DATA) == 0) & (first_spad_data_o != 0)) begin
                first_spad_words_o <= first_spad_addr_o + 1;
                current_state      <= LOADING_DATA;
                address_temp       <= 0;
                cycle_counter      <= 0;
              end
            end

            if (cycle_counter == (FIRST_SPAD_DATA_CYCLE - 1)) begin
              cycle_counter <= 0;
            end
            if ((address_temp == FIRST_SPAD_ADDR - 1) |((address_temp + 1) == first_spad_max_w)) begin
              first_spad_words_o <= first_spad_addr_o + 1;
              current_state      <= LOADING_DATA;
              address_temp       <= 0;
              cycle_counter      <= 0;
            end
          end else begin
            first_spad_en_o   <= 0;
            first_spad_addr_o <= 0;
          end
        end
        LOADING_DATA: begin  // Second state, start loading the second SPAD
          first_spad_en_o   <= 0;
          first_spad_addr_o <= 0;
          if (enable_i == 1) begin
            second_spad_en_o    <= 1;
            second_spad_addr_o  <= address_temp[SECOND_SPAD_ADDR_BITWIDTH-1:0];
            second_spad_words_o <= second_spad_addr_o + 2;

            address_temp        <= address_temp + 1;
            cycle_counter       <= cycle_counter + 1;

            data_storage        <= data_storage >> SECOND_SPAD_DATA;

            if (cycle_counter == 0) begin
              data_storage <= data_i;
            end

            if (cycle_counter == SECOND_SPAD_DATA_CYCLE - 1) begin
              cycle_counter <= 0;
            end
            if (address_temp == SECOND_SPAD_ADDR - 1) begin
              if (enable_i == 1) begin
                current_state <= LOADING_ADDR;
              end
              address_temp  <= 0;
              cycle_counter <= 0;
            end
          end else begin
            second_spad_en_o   <= 0;
            second_spad_addr_o <= 0;
          end
        end
        default: begin
          current_state <= LOADING_ADDR;  // Default state if none match
        end
      endcase
      if (compute_i) begin  // Reset SPAD addresses and state of module
        current_state       <= LOADING_ADDR;
        data_storage        <= 0;
        address_temp        <= 0;
        cycle_counter       <= 0;
        second_spad_words_o <= 0;
      end
    end
  end

endmodule
