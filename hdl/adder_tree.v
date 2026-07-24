`timescale 1ns / 1ps

module adder_tree #(
    parameter integer NUM_INPUTS    = 4,  // Number of input values
    parameter integer DATA_WIDTH    = 16  // Bit width of input and output data
)(
    input  wire                   clk_i,
    input  wire                   rst_ni,
    input  wire                   en_i,
    input  wire [NUM_INPUTS*DATA_WIDTH-1:0] data_i, // Packed vector of inputs
    
    output wire [DATA_WIDTH-1:0]  data_o,
    output wire                   valid_o
);

    // ------------------------------------------------------------------------
    // Helper functions / Localparams
    // ------------------------------------------------------------------------
    // Calculate required tree depth: $clog2(NUM_INPUTS)
    localparam integer STAGES = (NUM_INPUTS <= 1) ? 1 : $clog2(NUM_INPUTS);
    
    // Nearest power of 2 for symmetric binary tree structure
    localparam integer PAD_INPUTS = 1 << STAGES;

    // ------------------------------------------------------------------------
    // Pipeline Registers for Enable Signal
    // ------------------------------------------------------------------------
    reg [STAGES-1:0] valid_pipe_q;
    integer pipe_idx;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_pipe_q <= {STAGES{1'b0}};
        end else begin
            valid_pipe_q[0] <= en_i;
            for (pipe_idx = 1; pipe_idx < STAGES; pipe_idx = pipe_idx + 1) begin
                valid_pipe_q[pipe_idx] <= valid_pipe_q[pipe_idx-1];
            end
        end
    end

    assign valid_o = valid_pipe_q[STAGES-1];

    // ------------------------------------------------------------------------
    // Adder Tree Generation
    // ------------------------------------------------------------------------
    wire [DATA_WIDTH-1:0] stage_data [0:STAGES][0:PAD_INPUTS-1];

    // --- Input Stage Mapping & Padding ---
    genvar i;
    generate
        for (i = 0; i < PAD_INPUTS; i = i + 1) begin : g_input_map
            if (i < NUM_INPUTS) begin : g_real_input
                assign stage_data[0][i] = data_i[i*DATA_WIDTH +: DATA_WIDTH];
            end else begin : g_padded_input
                assign stage_data[0][i] = {DATA_WIDTH{1'b0}};
            end
        end
    endgenerate

    // --- Tree Construction ---
    genvar stage, pair;
    generate
        for (stage = 0; stage < STAGES; stage = stage + 1) begin : g_stages
            localparam integer NUM_ADDERS = PAD_INPUTS >> (stage + 1);
            
            // Derive enable signal safely for stage 0 and higher stages
            wire current_en = (stage == 0) ? en_i : valid_pipe_q[stage-1];

            for (pair = 0; pair < NUM_ADDERS; pair = pair + 1) begin : g_adders
                
                adder #(
                    .DATA_WIDTH_SUM(DATA_WIDTH)
                ) u_adder (
                    .clk_i       (clk_i),
                    .rst_ni      (rst_ni),
                    .summand_1_i (stage_data[stage][2*pair]),
                    .summand_2_i (stage_data[stage][2*pair + 1]),
                    .sum_o       (stage_data[stage+1][pair]),
                    .adder_en_i  (current_en)
                );

            end
        end
    endgenerate

    // --- Output Assignment ---
    assign data_o = stage_data[STAGES][0];

endmodule