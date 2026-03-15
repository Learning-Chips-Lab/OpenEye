// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: RST_SYNC
///
/// The Reset Synchronizer (RST_SYNC) is a critical module in the OpenEye architecture
/// that ensures reliable reset signal propagation across clock domains. It prevents
/// metastability issues by properly synchronizing asynchronous reset signals to the
/// target clock domain.
///
/// Key Features:
/// - Metastability Prevention:
///   * Multi-stage synchronization chain
///   * Configurable number of synchronizer stages
///   * Clean reset signal generation
///
/// - Implementation Flexibility:
///   * Synthesizable RTL description
///   * Technology mapping support
///   * Optional transparent mode
///
/// - Design Considerations:
///   * Active-low reset signals
///   * Asynchronous assertion
///   * Synchronous deassertion
///
/// Architectural Role:
/// The RST_SYNC module provides:
/// 1. Safe clock domain crossing for reset signals
/// 2. Consistent reset timing across the system
/// 3. Reliable system initialization
/// 4. Protection against reset glitches
///
/// Implementation Notes:
/// - Use this module for all reset synchronization needs
/// - Avoid direct reset synchronization elsewhere
/// - Module may be mapped to dedicated hardware cells
/// - Transparent mode available via OPENEYE_RST_SYNC_TRANSPARENT
///
/// Parameters:
///    NumStages         - Synchronizer Configuration
///                        Number of synchronizer flip-flop stages
///                        Default: 2 stages (recommended minimum)
///                        Larger values increase reliability but add latency
///   
/// Ports:
/// Clock Interface:
///    clk_i            - System Clock Input
///                       Target clock domain
///                       Positive edge triggered
///                       Synchronizer reference clock
///
/// Reset Interface:
///    rst_ni           - Asynchronous Reset Input (active low)
///                       External reset signal
///                       Can be asynchronous to clk_i
///                       Immediate assertion
///
///    rst_no           - Synchronized Reset Output (active low)
///                       Clean, synchronized reset
///                       Synchronous to clk_i
///                       Safe for use in target domain
///

module RST_SYNC #(
    // number of stages fixed to 2 for now...
    parameter NumStages = 2
) (
    input  clk_i,
    input  rst_ni,
    output rst_no
);

`ifndef OPENEYE_RST_SYNC_TRANSPARENT
  reg [NumStages-1:0] sync_stages;

  assign rst_no = sync_stages[1];

  always @(posedge clk_i, negedge rst_ni) begin
    if (rst_ni == 1'b0) begin
      sync_stages <= 0;
    end else begin
      sync_stages <= {sync_stages[0], 1'b1};
    end
  end

`else
  assign rst_no = rst_ni;
`endif


endmodule
