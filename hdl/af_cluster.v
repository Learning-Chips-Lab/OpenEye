// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1

`timescale 1ns / 1ps

/// Module: af_cluster
///
/// Overview:
/// A small activation-function helper block that can either pass data through
/// unchanged or apply a ReLU operation to it. The module stores the processed
/// value in a one-dimensional register array and emits the delayed value on the
/// following clock edge.
///
/// Parameters:
///   WIDTH - Bit width of each data element.
///   DEPTH - Number of entries in the internal one-dimensional delay array.
///   VECTOR_SIZE - Number of parallel input/output elements.
///
/// Ports:
///   clk_i      - Rising-edge clock input.
///   rst_ni     - Active-low reset input.
///   relu_en_i  - When 1, apply ReLU element-wise; when 0, pass data through.
///   data_i     - Vector of input data words.
///   data_o     - Vector of output data words delayed by the internal pipeline.

module af_cluster #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 2,
    parameter integer VECTOR_SIZE = 4
) (
    input  wire                      clk_i,
    input  wire                      rst_ni,
    input  wire                      relu_en_i,
    input  wire signed [WIDTH-1:0]   data_i [0:VECTOR_SIZE-1],
    output wire signed [WIDTH-1:0]   data_o [0:VECTOR_SIZE-1]
);

    // One-dimensional register array used as a simple shift register for each
    // vector element. Each entry stores one delayed sample, so the outputs are
    // available on the next clock cycle after the inputs are captured.
    reg signed [WIDTH-1:0] pipeline_reg [0:DEPTH-1][0:VECTOR_SIZE-1];

    integer idx;
    integer lane;

    // Main sequential logic.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            // Clear the pipeline on reset.
            for (idx = 0; idx < DEPTH; idx = idx + 1) begin
                for (lane = 0; lane < VECTOR_SIZE; lane = lane + 1) begin
                    pipeline_reg[idx][lane] <= {WIDTH{1'b0}};
                end
            end
        end else begin
            // Shift the previous values through the pipeline for each lane.
            for (idx = DEPTH-1; idx > 0; idx = idx - 1) begin
                for (lane = 0; lane < VECTOR_SIZE; lane = lane + 1) begin
                    pipeline_reg[idx][lane] <= pipeline_reg[idx-1][lane];
                end
            end

            // Apply the optional ReLU operation element-wise before storing the
            // new samples.
            for (lane = 0; lane < VECTOR_SIZE; lane = lane + 1) begin
                if (relu_en_i) begin
                    if (data_i[lane] < 0) begin
                        pipeline_reg[0][lane] <= {WIDTH{1'b0}};
                    end else begin
                        pipeline_reg[0][lane] <= data_i[lane];
                    end
                end else begin
                    pipeline_reg[0][lane] <= data_i[lane];
                end
            end
        end
    end

    // Drive the outputs directly from the last pipeline stage.
    genvar gv;
    generate
        for (gv = 0; gv < VECTOR_SIZE; gv = gv + 1) begin : gen_data_o
            assign data_o[gv] = pipeline_reg[DEPTH-1][gv];
        end
    endgenerate

endmodule
