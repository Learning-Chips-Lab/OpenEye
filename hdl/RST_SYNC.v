// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RST_SYNC
///
/// This module synchronizes an asynchronous reset signal
/// Use this module instead of implementing synchronization directly
/// because we might want to map this to an integrated cell in the
/// physical flow.
///
/// Parameters:
///     NumStages: Number of stages in the synchronizer. Default is 2.
///
/// Ports:
///     clk_i: Clock input
///     rst_ni: Asynchronous reset input
///     rst_no: Synchronized reset output
///

module RST_SYNC 
#(
    // number of stages fixed to 2 for now...
    parameter NumStages = 2
) (
    input          clk_i,
    input          rst_ni,
    output         rst_no
);

`ifndef OPENEYE_RST_SYNC_TRANSPARENT
    reg [NumStages-1:0] sync_stages;

    assign rst_no = sync_stages[1];

    always_ff @(posedge clk_i, negedge rst_ni) begin
        if (rst_ni == 1'b0) begin
            sync_stages <= 0;
        end else begin
            sync_stages <= { sync_stages[0], 1'b1 };
        end
    end

`else
    assign rst_no = rst_ni;
`endif


endmodule