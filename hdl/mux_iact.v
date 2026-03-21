// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: mux_iact
///
/// The Input Activation Multiplexer (mux_iact) is a specialized multiplexing module in
/// the OpenEye architecture designed for managing input activation data paths. It implements
/// a configurable N-to-1 multiplexer with integrated control signal handling for input
/// activation flow control.
///
/// Key Features:
/// - Multi-Channel Design:
///   * Configurable number of input paths
///   * Packed array interfaces
///   * Independent control signals
///
/// - Mixed Data Handling:
///   * Wide data path for activations
///   * Single-bit control signals
///   * Broadcast capability
///
/// - Dynamic Selection:
///   * Runtime path selection
///   * Zero-state support
///   * Control signal routing
///
/// Architectural Role:
/// The mux_iact serves as:
/// 1. Input activation router
/// 2. Control signal distributor
/// 3. Data path coordinator
/// 4. Flow control manager
///
/// Operational Modes:
/// 1. Normal Selection (sel_i < I_COUNT):
///    - Routes selected activation path
///    - Forwards corresponding control signals
///    - Maintains flow control
///
/// 2. Zero State (sel_i = I_COUNT):
///    - Outputs zero activation data
///    - Clears control signals
///    - Preserves broadcast capability
///
/// Parameters:
///    WIDTH           - Activation Data Width
///                     Width of each activation data path
///                     Determines processing precision
///
///    I_COUNT        - Input Channel Count
///                     Number of input activation paths
///                     Defines multiplexing scale
///   
/// Ports:
/// Data Arrays:
///    a_i            - Activation Input Array [WIDTH*I_COUNT-1:0]
///                     Packed array of I_COUNT activation vectors
///                     Each vector is WIDTH bits wide
///
///    b_i            - Control Input Array [I_COUNT-1:0]
///                     Packed array of control signals
///                     One bit per input channel
///
///    c_o            - Broadcast Output Array [I_COUNT-1:0]
///                     Distributes c_i to all channels
///                     Maintains synchronization
///
/// Control Interface:
///    sel_i          - Path Selection [$clog2(I_COUNT+1)-1:0]
///                     Selects active input channel
///                     I_COUNT value triggers zero state
///
/// Selected Outputs:
///    a_o            - Selected Activation [WIDTH-1:0]
///                     Currently selected activation data
///                     Zero when sel_i = I_COUNT
///
///    b_o            - Selected Control
///                     Control signal for selected path
///                     Zero when sel_i = I_COUNT
///
/// Broadcast Input:
///    c_i            - Broadcast Control
///                     Single-bit control input
///                     Distributed to all channels via c_o
///

module mux_iact #(
    parameter WIDTH   = 20,
    parameter I_COUNT = 3
) (
    input      [    WIDTH*I_COUNT-1:0] a_i,
    input      [          I_COUNT-1:0] b_i,
    output reg [          I_COUNT-1:0] c_o,
    input      [$clog2(I_COUNT+1)-1:0] sel_i,
    output reg [            WIDTH-1:0] a_o,
    output reg                         b_o,
    input                              c_i
);
  integer j;
  genvar i;

  wire [WIDTH-1:0] a_w[0:(I_COUNT)-1];
  for (i = 0; i < I_COUNT; i = i + 1) begin
    assign a_w[i] = a_i[(WIDTH*(i+1))-1:(WIDTH*i)];
  end
  always @(*) begin : configure_mux
    a_o = a_w[sel_i];
    b_o = b_i[sel_i];
    if (sel_i == I_COUNT) begin
      a_o = 0;
      b_o = 0;
    end
    for (j = 0; j < I_COUNT; j = j + 1) begin
      c_o[j] = c_i;
    end
  end

endmodule
