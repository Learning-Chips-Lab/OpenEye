(* use_dsp = "yes" *) // Force usage of DSP-Ressources
module dsp_unit #(
    parameter DATA_WIDTH_FAC1 = 1,
    parameter DATA_WIDTH_FAC2 = 1,
    parameter DATA_WIDTH_ADDI = 20,
    parameter DATA_WIDTH_PROD = DATA_WIDTH_FAC1+DATA_WIDTH_FAC2
) (
    input                               clk_i,
    input                               rst_ni,
    input                               multi_en_i,
    input                               adder_en_i,
    input                               adder_sel_i,
    input signed  [DATA_WIDTH_FAC1-1:0] a_in,
    input signed  [DATA_WIDTH_FAC2-1:0] b_in,
    input signed  [DATA_WIDTH_ADDI-1:0] c_in,
    input signed  [DATA_WIDTH_ADDI-1:0] d_in,
    output signed [DATA_WIDTH_ADDI-1:0] p_out
);
    reg signed [DATA_WIDTH_PROD-1:0] m_reg;
    reg signed [DATA_WIDTH_ADDI-1:0] p_reg;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            m_reg <= 0;
            p_reg <= 0;
        end else begin
            if (multi_en_i) begin
                m_reg <= a_in * b_in;
            end else begin
                m_reg <= 0;
            end
            if (adder_en_i) begin
                p_reg <= (!adder_sel_i) ? (c_in + m_reg) : (c_in + d_in);
            end else begin
                p_reg <= 0;
            end
        end
    end
    assign p_out = p_reg;
endmodule