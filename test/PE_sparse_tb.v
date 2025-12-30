// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Module: PE_sparse_tb
//
// Testbench for the Processing Element (PE) module in the OpenEye accelerator.
// This testbench performs a complete 1D convolution operation with:
// - Input activations: [1, 2, 3]
// - Weights (filter):  [2, 3, 4]
// - Bias:              5
// - Expected output:   5 + (1*2 + 2*3 + 3*4) = 5 + 20 = 25
//
// The testbench demonstrates:
// - Data loading into SPAD memories using CSC (Compressed Sparse Column) format
// - Triggering MAC (Multiply-Accumulate) computation
// - Loading bias/partial sum values
// - Validating output results
//
// Based on the Eyeriss v2 architecture described in:
// "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks
// on Mobile Devices" (Chen et al., arXiv:1807.07928v2)
////////////////////////////////////////////////////////////////////////////////

module PE_sparse_tb;

  // ============================================================================
  // Parameters - Match PE default configuration
  // ============================================================================

  parameter IS_TOPLEVEL = 1;
  parameter SERIAL      = 0;
  parameter CREATE_VCD  = 1;  // Enable VCD generation for waveform viewing

  parameter PE_X = 0;
  parameter PE_Y = 0;

  parameter integer PARALLEL_MACS = 2;

  // Data bitwidths
  parameter integer DATA_IACT_BITWIDTH     = 8;
  parameter integer DATA_WGHT_BITWIDTH     = 8;
  parameter integer DATA_PSUM_BITWIDTH     = 20;
  parameter integer DATA_IACT_OVERHEAD     = 4;
  parameter integer DATA_WGHT_IGNORE_ZEROS = 4;

  // Memory sizes
  parameter integer IACT_DATA_ADDR = 16;
  parameter integer IACT_ADDR_ADDR = 9;
  parameter integer WGHT_DATA_ADDR = 96;
  parameter integer WGHT_ADDR_ADDR = 16;
  parameter integer PSUM_ADDR = 32;

  // Transfer bitwidths
  parameter integer TRANS_BITWIDTH_IACT = 24;
  parameter integer TRANS_BITWIDTH_WGHT = 24;
  parameter integer NUM_GLB_IACT = 3;

  // Local parameters (calculated)
  localparam integer TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * PARALLEL_MACS;
  localparam integer IACT_DATA_DATA = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD;
  localparam integer WGHT_DATA_DATA = (DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS;

  // Timing parameters
  parameter CLK_PERIOD = 10;  // 10ns clock period (100 MHz)

  // Test configuration
  parameter NUM_IACT = 3;  // Number of input activations
  parameter NUM_WGHT = 3;  // Number of weights
  parameter NUM_OUTPUTS = 1; // Number of output filters

  // ============================================================================
  // Signal Declarations
  // ============================================================================

  // Clock and Reset
  reg                                          clk_i;
  reg                                          rst_ni;

  // Input Activation Interface
  reg  [$clog2(NUM_GLB_IACT+1)-1:0]            iact_select_i;
  reg  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]  iact_data_i;
  reg  [NUM_GLB_IACT-1:0]                      iact_enable_i;
  wire [NUM_GLB_IACT-1:0]                      iact_ready_o;

  // Weight Interface
  reg  [TRANS_BITWIDTH_WGHT-1:0]               wght_data_i;
  reg                                          wght_enable_i;
  wire                                         wght_ready_o;

  // Partial Sum Interface
  reg  [TRANS_BITWIDTH_PSUM-1:0]               psum_data_i;
  reg                                          psum_enable_i;
  wire                                         psum_ready_o;
  wire [TRANS_BITWIDTH_PSUM-1:0]               psum_data_o;
  wire                                         psum_enable_o;
  reg                                          psum_ready_i;

  // Control Interface
  reg                                          compute_i;
  reg                                          enable_stream_i;
  reg  [11:0]                                  data_stream_i;

  // ============================================================================
  // Test Variables
  // ============================================================================

  integer i, j, k;
  integer test_errors;
  integer cycle_count;

  // Test data arrays - Convolution test
  reg [DATA_IACT_BITWIDTH-1:0] test_iact [0:NUM_IACT-1];
  reg [DATA_WGHT_BITWIDTH-1:0] test_wght [0:NUM_WGHT-1];
  reg [DATA_PSUM_BITWIDTH-1:0] test_bias [0:NUM_OUTPUTS-1];
  reg [DATA_PSUM_BITWIDTH-1:0] expected_output [0:NUM_OUTPUTS-1];
  reg [DATA_PSUM_BITWIDTH-1:0] actual_output [0:NUM_OUTPUTS-1];

  // SPAD data format arrays
  reg [23:0] iact_addr_data [0:IACT_ADDR_ADDR-1];
  reg [23:0] iact_data_array [0:IACT_DATA_ADDR-1];
  reg [23:0] wght_addr_data [0:WGHT_ADDR_ADDR-1];
  reg [23:0] wght_data_array [0:WGHT_DATA_ADDR-1];

  // Output monitoring
  integer output_count;
  reg output_valid;

  // ============================================================================
  // DUT Instantiation
  // ============================================================================

  PE #(
    .IS_TOPLEVEL(IS_TOPLEVEL),
    .SERIAL(SERIAL),
    .CREATE_VCD(CREATE_VCD),
    .PE_X(PE_X),
    .PE_Y(PE_Y),
    .PARALLEL_MACS(PARALLEL_MACS),
    .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
    .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),
    .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
    .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
    .DATA_WGHT_IGNORE_ZEROS(DATA_WGHT_IGNORE_ZEROS),
    .IACT_DATA_ADDR(IACT_DATA_ADDR),
    .IACT_ADDR_ADDR(IACT_ADDR_ADDR),
    .WGHT_DATA_ADDR(WGHT_DATA_ADDR),
    .WGHT_ADDR_ADDR(WGHT_ADDR_ADDR),
    .PSUM_ADDR(PSUM_ADDR),
    .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
    .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
    .NUM_GLB_IACT(NUM_GLB_IACT)
  ) dut (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .iact_select_i(iact_select_i),
    .iact_data_i(iact_data_i),
    .iact_enable_i(iact_enable_i),
    .iact_ready_o(iact_ready_o),
    .wght_data_i(wght_data_i),
    .wght_enable_i(wght_enable_i),
    .wght_ready_o(wght_ready_o),
    .psum_data_i(psum_data_i),
    .psum_enable_i(psum_enable_i),
    .psum_ready_o(psum_ready_o),
    .psum_data_o(psum_data_o),
    .psum_enable_o(psum_enable_o),
    .psum_ready_i(psum_ready_i),
    .compute_i(compute_i),
    .enable_stream_i(enable_stream_i),
    .data_stream_i(data_stream_i)
  );

  // ============================================================================
  // Clock Generation
  // ============================================================================

  initial begin
    clk_i = 0;
    forever #(CLK_PERIOD/2) clk_i = ~clk_i;
  end

  // ============================================================================
  // VCD Dump for Waveform Viewing
  // ============================================================================

  initial begin
    if (CREATE_VCD) begin
      $dumpfile("PE_sparse_tb.vcd");
      $dumpvars(0, PE_sparse_tb);
    end
  end

  // ============================================================================
  // Helper Tasks
  // ============================================================================

  // Task: Reset all signals
  task reset_signals;
    begin
      rst_ni = 0;
      iact_select_i = 0;
      iact_data_i = 0;
      iact_enable_i = 0;
      wght_data_i = 0;
      wght_enable_i = 0;
      psum_data_i = 0;
      psum_enable_i = 0;
      psum_ready_i = 0;
      compute_i = 0;
      enable_stream_i = 0;
      data_stream_i = 0;
    end
  endtask

  // Task: Apply reset sequence
  task apply_reset;
    begin
      $display("[%0t] Applying reset...", $time);
      rst_ni = 0;
      #(CLK_PERIOD * 2);
      rst_ni = 1;
      #(CLK_PERIOD);
      $display("[%0t] Reset released", $time);
    end
  endtask

  // Task: Load input activations into SPAD
  // Format: CSC (Compressed Sparse Column) with overhead encoding
  task load_input_activations;
    integer addr_idx, data_idx;
    begin
      $display("[%0t] Loading input activations...", $time);

      // Enable IACT interface
      iact_enable_i = 3'b001;  // Enable first channel

      // Send address array first
      // For simple case: addr_data[0] = 3 (we have 3 activations)
      @(posedge clk_i);
      iact_addr_data[0] = 24'd3;  // Number of non-zero elements
      iact_addr_data[1] = 24'd3;  // Cumulative count
      iact_data_i = {48'h0, iact_addr_data[0]};  // Pack into transfer width

      @(posedge clk_i);
      iact_data_i = {48'h0, iact_addr_data[1]};

      // Send data array
      // Format: [overhead (4-bit) | data (8-bit)] = 12 bits per element
      // overhead = number of zeros skipped before this value
      @(posedge clk_i);
      iact_data_array[0] = {16'h0, 4'h0, test_iact[0]};  // 0 zeros skipped, value=1
      iact_data_i = {48'h0, iact_data_array[0]};

      @(posedge clk_i);
      iact_data_array[1] = {16'h0, 4'h0, test_iact[1]};  // 0 zeros skipped, value=2
      iact_data_i = {48'h0, iact_data_array[1]};

      @(posedge clk_i);
      iact_data_array[2] = {16'h0, 4'h0, test_iact[2]};  // 0 zeros skipped, value=3
      iact_data_i = {48'h0, iact_data_array[2]};

      // Deassert enable
      @(posedge clk_i);
      iact_enable_i = 3'b000;
      iact_data_i = 0;

      $display("[%0t] Input activations loaded", $time);
    end
  endtask

  // Task: Load weights into SPAD
  task load_weights;
    integer addr_idx, data_idx;
    reg [11:0] wght_with_overhead;
    begin
      $display("[%0t] Loading weights...", $time);

      // Enable weight interface
      wght_enable_i = 1'b1;

      // Send address array
      // Address array points to start of each filter's weights
      @(posedge clk_i);
      wght_addr_data[0] = 24'd0;  // First filter starts at index 0
      wght_data_i = wght_addr_data[0];

      @(posedge clk_i);
      wght_addr_data[1] = 24'd2;  // End marker (2 because PARALLEL_MACS=2, so 3 weights / 2 = ceil(1.5) = 2)
      wght_data_i = wght_addr_data[1];

      @(posedge clk_i);
      wght_addr_data[2] = 24'd2;  // Repeat for single filter
      wght_data_i = wght_addr_data[2];

      // Send data array in packed format (2 weights per 24-bit word)
      // Format: [overhead (4-bit) | weight (8-bit)] = 12 bits per weight
      // Pack 2 weights per transfer
      @(posedge clk_i);
      wght_with_overhead = {4'h0, test_wght[0]};  // Weight 0: overhead=0, value=2
      wght_data_array[0] = {wght_with_overhead, 4'h0, test_wght[1]};  // Weight 1: overhead=0, value=3
      wght_data_i = wght_data_array[0];

      @(posedge clk_i);
      wght_with_overhead = {4'h0, test_wght[2]};  // Weight 2: overhead=0, value=4
      wght_data_array[1] = {wght_with_overhead, 12'h0};  // Pad second half
      wght_data_i = wght_data_array[1];

      // Deassert enable
      @(posedge clk_i);
      wght_enable_i = 1'b0;
      wght_data_i = 0;

      $display("[%0t] Weights loaded", $time);
    end
  endtask

  // Task: Trigger computation
  task trigger_compute;
    begin
      $display("[%0t] Triggering computation...", $time);
      @(posedge clk_i);
      compute_i = 1'b1;
      @(posedge clk_i);
      compute_i = 1'b0;
      $display("[%0t] Compute signal sent", $time);
    end
  endtask

  // Task: Load bias/partial sum values
  task load_bias;
    begin
      $display("[%0t] Waiting for PE to request bias...", $time);

      // Set ready to receive output
      psum_ready_i = 1'b1;

      // Wait for PE to signal ready for psum input
      wait(psum_ready_o == 1'b1);
      @(posedge clk_i);

      $display("[%0t] Loading bias values...", $time);

      // Enable psum input
      psum_enable_i = 1'b1;

      // Send bias data (packed, 2 values per word for PARALLEL_MACS=2)
      // Format: [psum1 (20-bit) | psum0 (20-bit)] = 40 bits
      @(posedge clk_i);
      psum_data_i = {test_bias[0], 20'h0};  // First output bias

      @(posedge clk_i);
      psum_enable_i = 1'b0;
      psum_data_i = 0;

      $display("[%0t] Bias values loaded", $time);
    end
  endtask

  // Task: Monitor and validate outputs
  task monitor_outputs;
    integer timeout_cycles;
    begin
      $display("[%0t] Monitoring outputs...", $time);

      output_count = 0;
      timeout_cycles = 0;

      // Wait for output to start
      while (psum_enable_o == 1'b0 && timeout_cycles < 100) begin
        @(posedge clk_i);
        timeout_cycles = timeout_cycles + 1;
      end

      if (timeout_cycles >= 100) begin
        $display("[%0t] ERROR: Timeout waiting for output", $time);
        test_errors = test_errors + 1;
      end else begin
        $display("[%0t] Output started", $time);

        // Capture outputs while enabled
        while (psum_enable_o == 1'b1) begin
          // Extract two partial sums from 40-bit output
          actual_output[0] = psum_data_o[DATA_PSUM_BITWIDTH-1:0];  // Lower 20 bits
          if (PARALLEL_MACS > 1) begin
            actual_output[1] = psum_data_o[2*DATA_PSUM_BITWIDTH-1:DATA_PSUM_BITWIDTH];  // Upper 20 bits
          end

          $display("[%0t] Output received: psum[0]=%0d, psum[1]=%0d",
                   $time, actual_output[0], actual_output[1]);

          output_count = output_count + 1;
          @(posedge clk_i);
        end

        $display("[%0t] Output complete, received %0d values", $time, output_count);
      end
    end
  endtask

  // Task: Validate results
  task validate_results;
    reg result_ok;
    begin
      $display("================================================================================");
      $display("Result Validation");
      $display("================================================================================");

      result_ok = 1'b1;

      // Check first output
      if (actual_output[0] == expected_output[0]) begin
        $display("PASS: Output[0] = %0d (expected %0d)", actual_output[0], expected_output[0]);
      end else begin
        $display("FAIL: Output[0] = %0d (expected %0d)", actual_output[0], expected_output[0]);
        test_errors = test_errors + 1;
        result_ok = 1'b0;
      end

      if (result_ok) begin
        $display("*** CONVOLUTION TEST PASSED ***");
      end else begin
        $display("*** CONVOLUTION TEST FAILED ***");
      end

      $display("================================================================================");
    end
  endtask

  // ============================================================================
  // Test Stimulus - Main Convolution Test
  // ============================================================================

  initial begin
    // Initialize
    test_errors = 0;
    output_count = 0;
    cycle_count = 0;

    // Initialize arrays
    for (i = 0; i < IACT_ADDR_ADDR; i = i + 1) iact_addr_data[i] = 0;
    for (i = 0; i < IACT_DATA_ADDR; i = i + 1) iact_data_array[i] = 0;
    for (i = 0; i < WGHT_ADDR_ADDR; i = i + 1) wght_addr_data[i] = 0;
    for (i = 0; i < WGHT_DATA_ADDR; i = i + 1) wght_data_array[i] = 0;

    // Setup test data for 1D convolution
    // Input:   [1, 2, 3]
    // Filter:  [2, 3, 4]
    // Bias:    5
    // Result:  5 + (1*2 + 2*3 + 3*4) = 5 + (2 + 6 + 12) = 5 + 20 = 25
    test_iact[0] = 8'd1;
    test_iact[1] = 8'd2;
    test_iact[2] = 8'd3;
    test_wght[0] = 8'd2;
    test_wght[1] = 8'd3;
    test_wght[2] = 8'd4;
    test_bias[0] = 20'd5;
    expected_output[0] = 20'd25;

    // Reset signals
    reset_signals();

    // Print test banner
    $display("================================================================================");
    $display("PE Convolution Test");
    $display("================================================================================");
    $display("Clock Period: %0d ns", CLK_PERIOD);
    $display("");
    $display("Convolution Configuration:");
    $display("  Input Activations: [%0d, %0d, %0d]", test_iact[0], test_iact[1], test_iact[2]);
    $display("  Filter Weights:    [%0d, %0d, %0d]", test_wght[0], test_wght[1], test_wght[2]);
    $display("  Bias:              %0d", test_bias[0]);
    $display("");
    $display("Expected Computation:");
    $display("  Output = Bias + (I[0]*W[0] + I[1]*W[1] + I[2]*W[2])");
    $display("         = %0d + (%0d*%0d + %0d*%0d + %0d*%0d)",
             test_bias[0], test_iact[0], test_wght[0], test_iact[1], test_wght[1], test_iact[2], test_wght[2]);
    $display("         = %0d + (%0d + %0d + %0d)",
             test_bias[0], test_iact[0]*test_wght[0], test_iact[1]*test_wght[1], test_iact[2]*test_wght[2]);
    $display("         = %0d", expected_output[0]);
    $display("================================================================================");
    $display("");

    // Apply reset
    apply_reset();

    // Wait a few cycles
    repeat(5) @(posedge clk_i);

    // ========================================================================
    // Execute Convolution Test Sequence
    // ========================================================================

    $display("[%0t] ===== Starting Convolution Test Sequence =====", $time);
    $display("");

    // Step 1: Load input activations and weights in parallel
    fork
      load_input_activations();
      load_weights();
    join

    // Wait for data to settle
    repeat(5) @(posedge clk_i);

    // Step 2: Trigger computation
    trigger_compute();

    // Step 3: Load bias when PE requests it
    load_bias();

    // Step 4: Monitor outputs
    fork
      monitor_outputs();
    join

    // Wait for completion
    repeat(10) @(posedge clk_i);

    // ========================================================================
    // Validate Results
    // ========================================================================

    validate_results();

    // ========================================================================
    // Test Summary
    // ========================================================================

    $display("");
    $display("================================================================================");
    $display("Test Summary");
    $display("================================================================================");
    $display("Total Errors: %0d", test_errors);

    if (test_errors == 0) begin
      $display("*** ALL TESTS PASSED ***");
    end else begin
      $display("*** TEST FAILED - %0d ERRORS ***", test_errors);
    end

    $display("================================================================================");

    // Finish simulation
    #(CLK_PERIOD * 10);
    $finish;
  end

  // ============================================================================
  // Output Monitor - Real-time display
  // ============================================================================

  always @(posedge clk_i) begin
    if (psum_enable_o) begin
      $display("[%0t] PE Output Active: psum_data_o[39:20]=%0d, psum_data_o[19:0]=%0d",
               $time,
               psum_data_o[2*DATA_PSUM_BITWIDTH-1:DATA_PSUM_BITWIDTH],
               psum_data_o[DATA_PSUM_BITWIDTH-1:0]);
    end
  end

  // ============================================================================
  // Cycle Counter
  // ============================================================================

  always @(posedge clk_i) begin
    if (rst_ni) begin
      cycle_count = cycle_count + 1;
    end
  end

  // ============================================================================
  // Timeout Watchdog
  // ============================================================================

  initial begin
    #(CLK_PERIOD * 100);  // 100 cycle timeout
    $display("");
    $display("================================================================================");
    $display("ERROR: Simulation timeout after %0d cycles!", cycle_count);
    $display("================================================================================");
    $finish;
  end

endmodule
