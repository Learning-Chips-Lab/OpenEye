`timescale 1ns / 1ps

module iact_stream_constructor
#(
  // TODO wght_size changeable
  parameter   WGHT_SIZE           = 3,
  parameter   PADDING             = 1,

  parameter   N_PSUM              = 64,
  parameter   RAM_CELLS_X         = 8,
  parameter   RAM_CELLS_Y         = 4,
  parameter   RAM_CELLS_WORDWIDTH = 64,
  parameter   WORD_BITWIDTH       = 72,
  parameter   ADDRWIDTH           = 1,
  localparam  RAM_CELLS           = RAM_CELLS_X * RAM_CELLS_Y,
  localparam  PARAMS_SIZE         = 32,
  localparam  PARAM_LENGTH        = 8

) (
  input                                    clk_i,
  input                                    rst_ni,
  input      [PARAMS_SIZE-1:0]             params,
  input                                    enable_config,
  input                                    enable_converter,
  input      [RAM_CELLS_WORDWIDTH-1:0]     storage_i [RAM_CELLS_X-1:0][RAM_CELLS_Y-1:0],
  output reg                               ready_o,
  output reg                               mem_en_o,
  output reg [ADDRWIDTH-1:0]               mem_addr_o,
  output reg [WORD_BITWIDTH-1:0]           mem_data_o
);

  reg [PARAM_LENGTH-1:0]  x;
  reg [PARAM_LENGTH-1:0]  xx;
  reg [PARAM_LENGTH-1:0]  y;
  reg [PARAM_LENGTH-1:0]  yy;
  reg [PARAM_LENGTH-1:0]  ch;
  reg [PARAM_LENGTH-1:0]  ch_1;
  reg [PARAM_LENGTH-1:0]  ch_2;

  reg [PARAM_LENGTH-1:0]  iact_size;
  reg [PARAM_LENGTH-1:0]  channels;
  reg [PARAM_LENGTH-1:0]  max_cycles;
  reg [PARAM_LENGTH-1:0]  current_cycle;

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

  reg [15:0] pos;

  reg        n_en_reg;

  reg        cur_y_off_1;
  reg        cur_y_off_2;
  reg [4:0]  channel_offset_1;
  reg [4:0]  channel_offset_2;

  reg        change_state;

  typedef enum logic [1:0] {
    INITIALIZE        = 0,
    GET_PARAMETER     = 1,
    CONSTRUCTOR_READY = 2,
    WRITE_TO_MEMORY   = 3
  } state_t;

  reg  [31:0] fsm_cycle;
  state_t fsm_current_state;

  always @(posedge clk_i, negedge rst_ni) begin
    // Reset
    if (!rst_ni) begin
      fsm_cycle         <= 0;
      fsm_current_state <= INITIALIZE;
      mem_addr_o        <= 0;
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

      ready_o           <= 0;
      mem_en_o          <= 0;
      mem_addr_o        <= 0;
      mem_data_o        <= 0;
      max_cycles        <= 0;
      current_cycle     <= 0;


    end else begin
      case (fsm_current_state)
        INITIALIZE : begin
          fsm_cycle         <= 0;
          fsm_current_state <= GET_PARAMETER;
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

          ready_o           <= 0;
          mem_en_o          <= 0;
          mem_addr_o        <= 0;
          mem_data_o        <= 0;
        end

        GET_PARAMETER : begin
          if (enable_config == 1) begin
            x         <= params[PARAMS_SIZE-1:3*PARAMS_SIZE/4];
            y         <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4];
            iact_size <= params[(2*PARAMS_SIZE/4)-1:PARAMS_SIZE/4];
            channels  <= params[(PARAMS_SIZE/4)-1:0];
            fsm_cycle <= 1;
          end

          if (fsm_cycle >= 1) begin
            if (tmp_offset >= iact_size) begin
              tmp_offset <= tmp_offset - iact_size;
              y_offset   <= y_offset + 1;
            end else begin
              x_offset          <= tmp_offset;
              tmp_offset        <= N_PSUM;
              fsm_cycle         <= 0;
              max_cycles        <= iact_size + 2 * PADDING;
              fsm_current_state <= CONSTRUCTOR_READY;
            end
          end
        end

        CONSTRUCTOR_READY : begin
          ready_o <= 1;
          if (enable_converter == 1) begin
            fsm_current_state <= WRITE_TO_MEMORY;
          end
          mem_addr_o <= ~0;
        end

        WRITE_TO_MEMORY : begin
          fsm_cycle <= fsm_cycle + 1;
          mem_en_o  <= 0;
          if (fsm_cycle >= 1) begin
            mem_en_o  <= 1;
          end
          if (fsm_cycle % 2 == 0) begin
            mem_addr_o <= mem_addr_o + 1;
          end
          if (fsm_cycle == 2 * 4 * WGHT_SIZE) begin  //Router_cycle, 1 channels
            fsm_cycle <= 0;
            current_cycle <= current_cycle + 1;
            if (current_cycle == max_cycles - 1) begin
              fsm_current_state <= GET_PARAMETER;
            end
          end
        end

      endcase
    end
  end

endmodule