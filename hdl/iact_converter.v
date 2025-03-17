`timescale 1ns / 1ps

module iact_converter
#(
  // TODO wght_size changeable
  parameter WGHT_SIZE = 3,
  parameter N_PSUM    = 64
) (
  input             clk_i,
  input             rst_ni,

  input      [31:0] params,
  input             enable_config,
  input             enable_converter,
  output reg        ready,

  output reg        n_en_o,
  output reg [2:0]  n_o,
  output reg [3:0]  nx_o,
  output reg [10:0] mem_addr_o,
  output reg [3:0]  mem_off_o
);

  reg [7:0]  x;
  reg [7:0]  y;
  reg [7:0]  ch;

  reg [7:0]  iact_size;
  reg [7:0]  channels;

  reg [6:0]  x_offset;
  reg [6:0]  y_offset;
  reg [4:0]  row_cnt;
  reg [4:0]  y_offset_a;
  reg [4:0]  y_offset_b;
  reg [4:0]  channel_offset;

  reg        x_off_en;

  reg [1:0]  cur_x_off;
  reg [0:0]  cur_y_off;

  reg [7:0]  tmp_offset;
  reg        next_cycle;

  reg [7:0]  xx;
  reg [7:0]  yy;

  reg [15:0] pos;

  reg        n_en_reg;

  reg        cur_y_off_1;
  reg        cur_y_off_2;
  reg [4:0]  channel_offset_1;
  reg [4:0]  channel_offset_2;
  reg [7:0]  ch_1;
  reg [7:0]  ch_2;

  reg        change_state;

  typedef enum logic [1:0] {
    INITIALIZE      = 0,
    GET_PARAMETER   = 1,
    CONVERTER_READY = 2,
    GET_MEM_ADDR    = 3
  } state_t;

  reg  [31:0] fsm_cycle;
  state_t fsm_current_state;

  always @(posedge clk_i, negedge rst_ni) begin
    // Reset
    if (!rst_ni) begin
      fsm_cycle         <= 0;
      fsm_current_state <= INITIALIZE;
      n_en_o            <= 0;
      n_o               <= 0;
      nx_o              <= 0;
      mem_addr_o        <= 0;
      mem_off_o         <= 0;
      ready             <= 0;
      x                 <= 0;
      y                 <= 0;
      ch                <= 0;
      iact_size         <= 0;
      channels          <= 0;
      x_offset          <= 0;
      y_offset          <= 0;
      row_cnt           <= 0;
      y_offset_a        <= 0;
      y_offset_b        <= 1;
      channel_offset    <= 0;
      x_off_en          <= 0;
      cur_x_off         <= 0;
      cur_y_off         <= 0;
      tmp_offset        <= N_PSUM;
      next_cycle        <= 0;
      xx                <= 0;
      yy                <= 0;
      pos               <= 0;
      n_en_reg          <= 0;
      cur_y_off_1       <= 0;
      cur_y_off_2       <= 0;
      channel_offset_1  <= 0;
      channel_offset_2  <= 0;
      ch_1              <= 0;
      ch_2              <= 0;
      change_state      <= 0;


    end else begin
      case (fsm_current_state)
        INITIALIZE : begin
          fsm_cycle         <= 0;
          fsm_current_state <= GET_PARAMETER;
          ready             <= 0;
          iact_size         <= 0;
          channels          <= 0;
          x                 <= 0;
          y                 <= 0;
          ch                <= 0;
          ch_1              <= 0;
          ch_2              <= 0;
          xx                <= 0;
          yy                <= 0;
          pos               <= 0;
          x_offset          <= 0;
          y_offset          <= 0;
          cur_x_off         <= 0;
          cur_y_off         <= 0;
          cur_y_off_1       <= 0;
          cur_y_off_2       <= 0;
          row_cnt           <= 0;
          y_offset_a        <= 0;
          y_offset_b        <= 1;
          channel_offset    <= 0;
          channel_offset_1  <= 0;
          channel_offset_2  <= 0;
          n_en_o            <= 0;
          n_o               <= 0;
          nx_o              <= 0;
          mem_addr_o        <= 0;
          mem_off_o         <= 0;
        end

        GET_PARAMETER : begin
          if (enable_config == 1) begin
            x         <= params[31:24];
            y         <= params[23:16];
            iact_size <= params[15:8];
            channels  <= params[7:0];
            fsm_cycle <= 1;
          end

          if (fsm_cycle >= 1) begin
            if (tmp_offset >= iact_size) begin
              tmp_offset <= tmp_offset - iact_size;
              y_offset   <= y_offset + 1;
            end else begin
              x_offset   <= tmp_offset;
              tmp_offset <= N_PSUM;

              fsm_cycle         <= 0;
              fsm_current_state <= CONVERTER_READY;
            end
          end
        end

        CONVERTER_READY : begin
          ready <= 1;
          if (enable_converter == 1) begin
            fsm_current_state <= GET_MEM_ADDR;
          end
        end

        GET_MEM_ADDR : begin
          fsm_cycle        <= fsm_cycle + 1;
          cur_y_off_2      <= cur_y_off_1;
          cur_y_off_1      <= cur_y_off;
          channel_offset_2 <= channel_offset_1;
          channel_offset_1 <= channel_offset;
          ch_2             <= ch_1;
          ch_1             <= ch;
          
          // delay state_change by one cycle
          if (y >= iact_size) begin
            change_state <= 1;
          end
          if (change_state == 1) begin
            fsm_current_state <= INITIALIZE;
            change_state      <= 0;
          end

          if (cur_x_off + 1 >= 3) begin
            cur_x_off <= 0;
            if (cur_y_off + 1 >= 2) begin
              cur_y_off <= 0;
              next_cycle = 1;
            end else begin
              cur_y_off <= cur_y_off + 1;
            end
          end else begin
            cur_x_off <= cur_x_off + 1;
          end

          if (fsm_cycle >= 0) begin
            xx <= x + cur_x_off;
            if (cur_y_off == 0) begin
              yy <= y + y_offset_a;
            end else begin
              yy <= y + y_offset_b;
            end
          end

          if (fsm_cycle >= 1) begin
            if (yy == 0 || yy > iact_size) begin
              n_en_reg <= 0;
            end else begin
              if (xx == 0 || xx > iact_size) begin
                n_en_reg <= 0;
              end else begin
                n_en_reg <= 1;
                pos <= (yy - 1) * iact_size + (xx - 1);
              end
            end
          end

          if (fsm_cycle >= 2) begin
            n_en_o <= n_en_reg;
            ready  <= 1;

            if (channel_offset_2 == 1 && cur_y_off_2 == 1) begin
              n_o <= (((pos >> 6) & 1) << 1) | ((ch_2+1) & 1);
            end else begin
              n_o <= (((pos >> 6) & 1) << 1) | ((ch_2) & 1);
            end
            nx_o <= (pos >> 3) & 7;
            mem_addr_o <= ((pos >> 7) * (channels >> 1)) + (ch_2 >> 1);
            mem_off_o <= pos & 7;
          end

          if (next_cycle == 1) begin
            next_cycle <= 0;
            if (row_cnt + 2 >= WGHT_SIZE) begin
              row_cnt <= row_cnt + 2 - WGHT_SIZE;

              if (ch == channels-1) begin
                ch <= 0;

                // TODO: only 3x3 kernel
                if (x_off_en == 1) begin
                  x_off_en <= 0;
                  
                  if (x - 3 + x_offset >= iact_size) begin
                    x <= x - 3 + x_offset - iact_size;
                    y <= y + 1 + y_offset;
                  end else begin
                    x <= x - 3 + x_offset;
                    y <= y + y_offset;
                  end
                  
                end else begin
                  x_off_en <= 1;
                  x        <= x + 3;
                end
              end else begin
                ch <= ch + 1;
              end
            end else begin
              row_cnt <= row_cnt + 2;
            end

            // TODO: only 3x3 kernel
            case (y_offset_a)
              0 : begin
                y_offset_a     <= 2;
                y_offset_b     <= 0;
                channel_offset <= 1;
              end

              1 : begin
                y_offset_a     <= 0;
                y_offset_b     <= 1;
                channel_offset <= 0;
              end

              2 : begin
                y_offset_a <= 1;
                y_offset_b <= 2;
                channel_offset <= 0;
              end

            endcase

          end

        end

      endcase
    end
  end

endmodule