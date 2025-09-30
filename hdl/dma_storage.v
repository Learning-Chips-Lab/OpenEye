// Auto-generated from regmap.yaml

`include "regmap_params.vh"

module dma_storage (
    input  wire                  clk_i,
    input  wire                  rst_ni,
    input  wire                  write_en,
    input  wire [$clog2(TRANSMISSIONS)-1:0] write_addr,
    input  wire [DMA_BITWIDTH-1:0] dma_data_i,
    output reg [7:0] wght_cycles_reg,
    output reg [2:0] stride_x_reg,
    output reg [2:0] stride_y_reg,
    output reg [0:0] skipIact_reg,
    output reg [0:0] skipWght_reg,
    output reg [0:0] skipPsum_reg,
    output reg [3:0] psum_delay_reg,
    output reg [3:0] kernel_per_pe_cluster_reg,
    output reg [3:0] kernel_size,
    output reg [7:0] x_lines_reg,
    output reg [7:0] needed_wght_cycles_reg,
    output reg [17:0] needed_cycles_reg,
    output reg [7:0] iact_converter_buffer_addr_max_cycles,
    output reg [7:0] iact_channels_per_pe,
    output reg [7:0] iact_size_y,
    output reg [7:0] iact_size_x,
    output reg [10:0] iact_needed_cycles,
    output reg [4:0] kernels_per_calc,
    output reg [3:0] y_lines_per_calc,
    output reg [7:0] output_cycles,
    output reg [0:0] store_in_psum,
    output reg [0:0] max_pooling,
    output reg [0:0] fully_connected_layer,
    output reg [0:0] choose_iact_buffer_output,
    output reg [0:0] choose_iact_buffer_input,
    output reg [3:0] iact_channels_per_pe_next_layer,
    output reg [7:0] needed_psum_storage_cycles_reg,
    output reg [7:0] iact_channel_max_cycles,
    output reg [4:0] input_activations_reg,
    output reg [5:0] filters_reg,
    output reg [1:0] needed_x_cls_reg,
    output reg [3:0] needed_y_cls_reg,
    output reg [3:0] needed_iact_cycles_reg,
    output reg [4:0] wght_addr_len_reg,
    output reg [3:0] iact_addr_len_reg,
    output reg [0:0] send_data_out
);

always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  ///Reset
        wght_cycles_reg <= 8'd0;
        stride_x_reg <= 3'd0;
        stride_y_reg <= 3'd0;
        skipIact_reg <= 1'd0;
        skipWght_reg <= 1'd0;
        skipPsum_reg <= 1'd0;
        psum_delay_reg <= 4'd0;
        kernel_per_pe_cluster_reg <= 4'd0;
        kernel_size <= 4'd0;
        x_lines_reg <= 8'd0;
        needed_wght_cycles_reg <= 8'd0;
        needed_cycles_reg <= 18'd0;
        iact_converter_buffer_addr_max_cycles <= 8'd0;
        iact_channels_per_pe <= 8'd0;
        iact_size_y <= 8'd0;
        iact_size_x <= 8'd0;
        iact_needed_cycles <= 11'd0;
        kernels_per_calc <= 5'd0;
        y_lines_per_calc <= 4'd0;
        output_cycles <= 8'd0;
        store_in_psum <= 1'd0;
        max_pooling <= 1'd0;
        fully_connected_layer <= 1'd0;
        choose_iact_buffer_output <= 1'd0;
        choose_iact_buffer_input <= 1'd0;
        iact_channels_per_pe_next_layer <= 4'd0;
        needed_psum_storage_cycles_reg <= 8'd0;
        iact_channel_max_cycles <= 8'd0;
        input_activations_reg <= 5'd0;
        filters_reg <= 6'd0;
        needed_x_cls_reg <= 2'd0;
        needed_y_cls_reg <= 4'd0;
        needed_iact_cycles_reg <= 4'd0;
        wght_addr_len_reg <= 5'd0;
        iact_addr_len_reg <= 4'd0;
        send_data_out <= 1'd0;
    end else if (write_en) begin
        case (write_addr)
            0: begin
                wght_cycles_reg <= dma_data_i[PARAMETER_POS_0_0 + 7 : PARAMETER_POS_0_0];
                stride_x_reg <= dma_data_i[PARAMETER_POS_0_1 + 2 : PARAMETER_POS_0_1];
                stride_y_reg <= dma_data_i[PARAMETER_POS_0_2 + 2 : PARAMETER_POS_0_2];
                skipIact_reg <= dma_data_i[PARAMETER_POS_0_3 + 0 : PARAMETER_POS_0_3];
                skipWght_reg <= dma_data_i[PARAMETER_POS_0_4 + 0 : PARAMETER_POS_0_4];
                skipPsum_reg <= dma_data_i[PARAMETER_POS_0_5 + 0 : PARAMETER_POS_0_5];
                psum_delay_reg <= dma_data_i[PARAMETER_POS_0_6 + 3 : PARAMETER_POS_0_6];
                kernel_per_pe_cluster_reg <= dma_data_i[PARAMETER_POS_0_7 + 3 : PARAMETER_POS_0_7];
                kernel_size <= dma_data_i[PARAMETER_POS_0_8 + 3 : PARAMETER_POS_0_8];
                x_lines_reg <= dma_data_i[PARAMETER_POS_0_9 + 7 : PARAMETER_POS_0_9];
                needed_wght_cycles_reg <= dma_data_i[PARAMETER_POS_0_10 + 7 : PARAMETER_POS_0_10];
                needed_cycles_reg <= dma_data_i[PARAMETER_POS_0_11 + 17 : PARAMETER_POS_0_11];
            end
            1: begin
                iact_converter_buffer_addr_max_cycles <= dma_data_i[PARAMETER_POS_1_0 + 7 : PARAMETER_POS_1_0];
                iact_channels_per_pe <= dma_data_i[PARAMETER_POS_1_1 + 7 : PARAMETER_POS_1_1];
                iact_size_y <= dma_data_i[PARAMETER_POS_1_2 + 7 : PARAMETER_POS_1_2];
                iact_size_x <= dma_data_i[PARAMETER_POS_1_3 + 7 : PARAMETER_POS_1_3];
                iact_needed_cycles <= dma_data_i[PARAMETER_POS_1_4 + 10 : PARAMETER_POS_1_4];
                kernels_per_calc <= dma_data_i[PARAMETER_POS_1_5 + 4 : PARAMETER_POS_1_5];
                y_lines_per_calc <= dma_data_i[PARAMETER_POS_1_6 + 3 : PARAMETER_POS_1_6];
                output_cycles <= dma_data_i[PARAMETER_POS_1_7 + 7 : PARAMETER_POS_1_7];
                store_in_psum <= dma_data_i[PARAMETER_POS_1_8 + 0 : PARAMETER_POS_1_8];
                max_pooling <= dma_data_i[PARAMETER_POS_1_9 + 0 : PARAMETER_POS_1_9];
                fully_connected_layer <= dma_data_i[PARAMETER_POS_1_10 + 0 : PARAMETER_POS_1_10];
                choose_iact_buffer_output <= dma_data_i[PARAMETER_POS_1_11 + 0 : PARAMETER_POS_1_11];
            end
            2: begin
                choose_iact_buffer_input <= dma_data_i[PARAMETER_POS_2_0 + 0 : PARAMETER_POS_2_0];
                iact_channels_per_pe_next_layer <= dma_data_i[PARAMETER_POS_2_1 + 3 : PARAMETER_POS_2_1];
                needed_psum_storage_cycles_reg <= dma_data_i[PARAMETER_POS_2_2 + 7 : PARAMETER_POS_2_2];
                iact_channel_max_cycles <= dma_data_i[PARAMETER_POS_2_3 + 7 : PARAMETER_POS_2_3];
                input_activations_reg <= dma_data_i[PARAMETER_POS_2_4 + 4 : PARAMETER_POS_2_4];
                filters_reg <= dma_data_i[PARAMETER_POS_2_5 + 5 : PARAMETER_POS_2_5];
                needed_x_cls_reg <= dma_data_i[PARAMETER_POS_2_6 + 1 : PARAMETER_POS_2_6];
                needed_y_cls_reg <= dma_data_i[PARAMETER_POS_2_7 + 3 : PARAMETER_POS_2_7];
                needed_iact_cycles_reg <= dma_data_i[PARAMETER_POS_2_8 + 3 : PARAMETER_POS_2_8];
                wght_addr_len_reg <= dma_data_i[PARAMETER_POS_2_9 + 4 : PARAMETER_POS_2_9];
                iact_addr_len_reg <= dma_data_i[PARAMETER_POS_2_10 + 3 : PARAMETER_POS_2_10];
                send_data_out <= dma_data_i[PARAMETER_POS_2_11 + 0 : PARAMETER_POS_2_11];
            end
        endcase
    end
end

endmodule
