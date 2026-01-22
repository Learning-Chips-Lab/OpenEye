(* use_dsp = "yes" *) // Force usage of DSP-Ressources
module dsp_unit #(
    parameter DATA_WIDTH_FAC_1 = 1,
    parameter DATA_WIDTH_FAC_2 = 1,
    parameter DATA_WIDTH_ADDI = 20,
    parameter DATA_WIDTH_PROD = DATA_WIDTH_FAC_1+DATA_WIDTH_FAC_2
) (
    input wire                         clk,
    input wire                         rst,
    input wire  [DATA_WIDTH_FAC_1-1:0] a_in,
    input wire  [DATA_WIDTH_FAC_2-1:0] b_in,
    input wire  [DATA_WIDTH_ADDI-1:0]  c_in,
    output wire [DATA_WIDTH_ADDI-1:0]  p_out
);

    reg [DATA_WIDTH_FAC_1-1:0] a_reg;
    reg [DATA_WIDTH_FAC_2-1:0] b_reg;
    reg [ DATA_WIDTH_ADDI-1:0] c_reg;
    reg [ DATA_WIDTH_PROD-1:0] m_reg;
    reg [ DATA_WIDTH_ADDI-1:0] p_reg;

    always @(posedge clk) begin
        if (rst) begin
            a_reg <= 0;
            b_reg <= 0;
            c_reg <= 0;
            m_reg <= 0;
            p_reg <= 0;
        end else begin
            a_reg <= a_in;
            b_reg <= b_in;
            c_reg <= c_in;

            m_reg <= a_reg * b_reg;

            p_reg <= m_reg + c_reg;
        end
    end

    assign p_out = p_reg;

endmodule