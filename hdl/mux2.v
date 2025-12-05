// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: mux2
///
/// The 2-to-1 Multiplexer (mux2) is a fundamental building block in the OpenEye 
/// architecture that provides dynamic data path selection. This configurable module 
/// enables efficient data routing by selecting between two input sources based on 
/// a control signal.
///
/// Key Features:
/// - Binary Selection:
///   * Two input data paths
///   * Single-bit control
///   * Configurable data width
///
/// - Combinational Design:
///   * Zero-latency operation
///   * Glitch-free switching
///   * Direct signal routing
///
/// - Data Path Flexibility:
///   * Parameterized width
///   * Symmetric inputs
///   * Clean output generation
///
/// Architectural Role:
/// The mux2 serves as:
/// 1. Data path selector
/// 2. Source switching element
/// 3. Routing control point
/// 4. Signal multiplexing unit
///
/// Operational Modes:
/// 1. A-Input Selection (sel_i = 1):
///    - Routes a_in to output
///    - Ignores b_in
///
/// 2. B-Input Selection (sel_i = 0):
///    - Routes b_in to output
///    - Ignores a_in
///
/// Parameters:
///    DATA_WIDTH       - Data Path Configuration
///                      Width of input and output buses
///                      Defines signal granularity
///                      Typically matches system data width
///   
/// Ports:
/// Data Inputs:
///    a_in           - Primary Input [DATA_WIDTH-1:0]
///                     Selected when sel_i is 1
///                     Active data path for sel_i high
///
///    b_in           - Secondary Input [DATA_WIDTH-1:0]
///                     Selected when sel_i is 0
///                     Active data path for sel_i low
///
/// Control Interface:
///    sel_i          - Selection Control Input
///                     1: Select a_in
///                     0: Select b_in
///
/// Data Output:
///    y_o            - Multiplexed Output [DATA_WIDTH-1:0]
///                     Routes selected input
///                     Updates combinationally
////
module mux2 #(
    parameter DATA_WIDTH = 20
) (
    input  [DATA_WIDTH-1:0] a_in,
    input  [DATA_WIDTH-1:0] b_in,
    input                   sel_i,
    output [DATA_WIDTH-1:0] y_o
);

  assign y_o = sel_i ? a_in : b_in;

endmodule
