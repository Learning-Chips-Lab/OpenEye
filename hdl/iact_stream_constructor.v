`timescale 1ns / 1ps

module iact_stream_constructor
#(
  // TODO wght_size changeable
  parameter   WGHT_SIZE           = 3,
  parameter   PADDING             = 1,
  parameter   NEEDED_IACT_CYCLES  = 2,

  parameter   N_PSUM              = 64,

  parameter   NUM_GLB_IACT        = 3,
  parameter   DATA_IACT_BITWIDTH  = 8,
  parameter   DATA_IACT_OVERHEAD  = 4,
  parameter   RAM_CELLS           = 32,
  parameter   RAM_CELLS_WORDWIDTH = 64,
  parameter   WORD_BITWIDTH       = 72,
  parameter   ADDRWIDTH           = 1,
  localparam  IACT_WORDS_IN_RAM   = $floor(RAM_CELLS_WORDWIDTH/DATA_IACT_BITWIDTH),
  localparam  IACT_DATA_DATA      = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD,
  localparam  BITS_PER_ROUTER     = $floor(WORD_BITWIDTH/NUM_GLB_IACT),
  localparam  WORDS_PER_TRANS     = $floor(BITS_PER_ROUTER/IACT_DATA_DATA),
  localparam  PARAMS_SIZE         = 32,
  localparam  PARAM_LENGTH        = 8

) (
  input                                    clk_i,
  input                                    rst_ni,
  input      [PARAMS_SIZE-1:0]             params,
  input                                    enable_config,
  input                                    enable_converter,
  input      [RAM_CELLS_WORDWIDTH-1:0]     storage_i [RAM_CELLS-1:0],
  output reg                               ready_o,
  output reg                               mem_en_o,
  output reg [ADDRWIDTH-1:0]               mem_addr_o,
  output     [WORD_BITWIDTH-1:0]           mem_data_o
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

  reg [DATA_IACT_BITWIDTH-1:0] mem_data_payload_reg  [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  reg [DATA_IACT_OVERHEAD-1:0] mem_data_overhead_reg [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  wire [DATA_IACT_BITWIDTH-1:0] storage_w [RAM_CELLS-1:0][IACT_WORDS_IN_RAM-1:0];
  reg [PARAM_LENGTH-1:0] iact_router_counter;
  reg [PARAM_LENGTH-1:0] kernel_y_counter;

  reg [PARAM_LENGTH-1:0] w_var;
  reg [PARAM_LENGTH-1:0] x_var;
  reg [PARAM_LENGTH-1:0] y_var;
  reg [PARAM_LENGTH-1:0] ram_var;
  reg [PARAM_LENGTH-1:0] byte_var;
  typedef enum logic [1:0] {
    INITIALIZE        = 0,
    GET_PARAMETER     = 1,
    CONSTRUCTOR_READY = 2,
    WRITE_TO_MEMORY   = 3
  } state_t;

  generate
  reg  [31:0] fsm_cycle;
  state_t fsm_current_state;

  always @(posedge clk_i, negedge rst_ni) begin
    // Reset
    if (!rst_ni) begin
      fsm_cycle           <= 0;
      fsm_current_state   <= INITIALIZE;
      mem_addr_o          <= 0;
      x                   <= 0;
      y                   <= 0;
      ch                  <= 0;
      iact_size           <= 0;
      channels            <= 0;
      x_offset            <= 0;
      y_offset            <= 0;
      row_cnt             <= 0;
      y_offset_a          <= 0;
      y_offset_b          <= 1;
      channel_offset      <= 0;
      x_off_en            <= 0;
      cur_x_off           <= 0;
      cur_y_off           <= 0;
      tmp_offset          <= N_PSUM;
      next_cycle          <= 0;
      xx                  <= 0;
      yy                  <= 0;
      pos                 <= 0;
      n_en_reg            <= 0;
      cur_y_off_1         <= 0;
      cur_y_off_2         <= 0;
      channel_offset_1    <= 0;
      channel_offset_2    <= 0;
      ch_1                <= 0;
      ch_2                <= 0;
      change_state        <= 0;

      ready_o             <= 0;
      mem_en_o            <= 0;
      mem_addr_o          <= 0;
      max_cycles          <= 0;
      current_cycle       <= 0;
      iact_router_counter <= 0;
      kernel_y_counter    <= 0;
      w_var                = 0;
      x_var                = 0;
      y_var                = 0;
      ram_var              = 0;
      byte_var             = 0;
      for (int r=0; r<NUM_GLB_IACT; r=r+1) begin
        for (int w=0; w<WORDS_PER_TRANS; w=w+1) begin
          mem_data_payload_reg[r][w]  <= DATA_IACT_BITWIDTH'(0);
          mem_data_overhead_reg[r][w] <= DATA_IACT_OVERHEAD'(0);
        end
      end


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
        end

        GET_PARAMETER : begin
          mem_en_o            <= 0;
          mem_addr_o          <= 0;
          iact_router_counter <= 0;
          kernel_y_counter    <= 0;
          if (enable_config == 1) begin
            x         <= params[PARAMS_SIZE-1:3*PARAMS_SIZE/4];
            y         <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4];
            iact_size <= params[(2*PARAMS_SIZE/4)-1:PARAMS_SIZE/4];
            channels  <= params[(PARAMS_SIZE/4)-1:0];
            fsm_cycle <= 1;
          end

          if (fsm_cycle == 1) begin
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
        end

        WRITE_TO_MEMORY : begin
          fsm_cycle <= fsm_cycle + 1;
          mem_en_o  <= 0;

          if (fsm_cycle % 2 == 2 - 1) begin
            mem_en_o  <= 1;
          end
          if ((fsm_cycle+1) % (channels * NEEDED_IACT_CYCLES) == 0) begin
            y <= y + 1;
          end
          if ((fsm_cycle+1) % channels == 0) begin
              iact_router_counter <= iact_router_counter + 1;
              if (iact_router_counter == NEEDED_IACT_CYCLES - 1) begin
                iact_router_counter <= 0;
                kernel_y_counter    <= kernel_y_counter + 1;
              end
          end
          if (fsm_cycle % 2 == 0) begin
            if (fsm_cycle % channels != 0) begin
              mem_addr_o <= mem_addr_o + 1;
            end else begin
              mem_addr_o <= (kernel_y_counter * channels + iact_router_counter * (channels*WGHT_SIZE))/WORDS_PER_TRANS;
            end
            //Reset payload to 0
            for (int r=0; r<NUM_GLB_IACT; r++) begin
              for (int w=0; w<WORDS_PER_TRANS; w++) begin
                mem_data_payload_reg[r][w] <= 0;
              end
            end

            if (fsm_cycle % channels == 0) begin
              for (int r=0; r<NUM_GLB_IACT; r++) begin
                for (int w=0; w<WORDS_PER_TRANS; w++) begin
                  mem_data_overhead_reg[r][w] <= w;
                end
              end
            end else begin
              for (int r=0; r<NUM_GLB_IACT; r++) begin
                for (int w=0; w<WORDS_PER_TRANS; w++) begin
                  mem_data_overhead_reg[r][w] <= mem_data_overhead_reg[r][w] + 2;
                end
              end
            end
          end
          for (int r=0; r<NUM_GLB_IACT; r++) begin
            w_var = fsm_cycle % 2;
            x_var = (iact_router_counter * NUM_GLB_IACT) + (r + x);
            y_var = (y);
            ram_var = ((((y_var - PADDING)*iact_size) + (x_var-PADDING))/2)%RAM_CELLS;
            byte_var = ((((x_var - PADDING))*channels) + ((fsm_cycle%4)/2)* 2 + w_var)%IACT_WORDS_IN_RAM;
            //PADDING
            if ((
            ((PADDING) > x_var)|
            ((iact_size + PADDING - 1) < x_var)) | (
            ((PADDING) > y_var) |
            ((iact_size + PADDING - 1) < y_var)
            )) begin
              mem_data_payload_reg[r][w_var] <= 0;
            end else begin
              mem_data_payload_reg[r][w_var] <= storage_w[ram_var][byte_var];
            end
            w_var    = 0;
            x_var    = 0;
            y_var    = 0;
            byte_var = 0;
          end
          if (fsm_cycle == NEEDED_IACT_CYCLES * channels * WGHT_SIZE - 1) begin  //Router_cycle, 1 channels
            fsm_cycle <= 0;
            current_cycle <= current_cycle + 1;
            fsm_current_state <= GET_PARAMETER;
          end
        end

      endcase
    end
  end
  genvar r, w, b;
  for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
    for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
      for (b = 0; b < DATA_IACT_BITWIDTH; b = b + 1) begin
        localparam int index = w * IACT_DATA_DATA + r * WORDS_PER_TRANS * IACT_DATA_DATA + b;
        assign mem_data_o[index] = 
               mem_data_payload_reg[r][w][b];
      end
      for (b = 0; b < DATA_IACT_OVERHEAD; b = b + 1) begin
        localparam int index = r * WORDS_PER_TRANS * IACT_DATA_DATA + w * IACT_DATA_DATA + DATA_IACT_BITWIDTH + b;
        assign mem_data_o[index] = 
               mem_data_overhead_reg[r][w][b];
      end
    end
  end
  for (r = 0; r < RAM_CELLS; r = r + 1) begin
    for (w = 0; w < IACT_WORDS_IN_RAM; w = w + 1) begin
      for (b = 0; b < DATA_IACT_BITWIDTH; b = b + 1) begin
        localparam int index = b + w * DATA_IACT_BITWIDTH;
        assign storage_w[r][w][b] = storage_i[r][index];
      end
    end
  end

  endgenerate

endmodule