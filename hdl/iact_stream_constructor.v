`timescale 1ns / 1ps

module iact_stream_constructor #(
    // TODO wght_size changeable
    parameter  CALC_DATA_WIDTH     = 32,
    parameter  NUM_GLB_IACT        = 3,
    parameter  PE_X                = 4,
    parameter  PE_Y                = 3,
    parameter  DATA_IACT_BITWIDTH  = 8,
    parameter  DATA_IACT_OVERHEAD  = 4,
    parameter  RAM_CELLS           = 32,
    parameter  RAM_CELLS_WORDWIDTH = 64,
    parameter  WORD_BITWIDTH       = 72,
    parameter  ADDRWIDTH           = 12,
    localparam PES                 = PE_X * PE_Y,
    localparam IACT_WORDS_IN_RAM   = RAM_CELLS_WORDWIDTH / DATA_IACT_BITWIDTH,
    localparam IACT_DATA_DATA      = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD,
    localparam BITS_PER_ROUTER     = WORD_BITWIDTH / NUM_GLB_IACT,
    localparam WORDS_PER_TRANS     = BITS_PER_ROUTER / IACT_DATA_DATA,
    localparam PARAMS_SIZE         = 32,
    localparam PARAM_LENGTH        = 8,
    localparam WORDS_PER_CYCLE     = 2

) (
    input                                          clk_i,
    input                                          rst_ni,
    input                                          reset_cycle_i,
    input      [                  PARAMS_SIZE-1:0] params,
    input                                          enable_config,
    input                                          enable_store,
    input                                          enable_converter,
    input      [RAM_CELLS_WORDWIDTH*RAM_CELLS-1:0] storage_i,
    output reg                                     ready_o,
    input      [                 NUM_GLB_IACT-1:0] iact_ready_i,
    output reg [ NUM_GLB_IACT*BITS_PER_ROUTER-1:0] iact_data_o,
    output reg [                 NUM_GLB_IACT-1:0] iact_enable_o,
    output reg [ (PES*$clog2(NUM_GLB_IACT+1))-1:0] iact_choose_o,
    input      [                            4-1:0] needed_cycles_i,
    input      [                            8-1:0] iact_size_xi,
    input      [                            8-1:0] iact_size_yi,
    input      [                            8-1:0] x_lines_i
);
  reg                           ram_wr_en;
  reg  [         ADDRWIDTH-1:0] ram_wr_addr;
  wire [     WORD_BITWIDTH-1:0] ram_data_i;
  reg                           ram_rd_en;
  reg  [         ADDRWIDTH-1:0] ram_rd_addr;
  wire [     WORD_BITWIDTH-1:0] ram_data_o;
  reg  [         ADDRWIDTH-1:0] address_storage;
  reg  [                 8-1:0] x_lines_reg;
  reg  [                 8-1:0] x;
  reg  [                 8-1:0] y;
  reg  [                 8-1:0] channels;
  reg  [                 8-1:0] current_cycle;
  reg  [                 8-1:0] pos;
  reg  [                 3-1:0] padding_reg;
  reg  [                 4-1:0] needed_iact_cycles_reg;
  reg  [                 4-1:0] current_iact_cycle_reg;
  reg  [                 4-1:0] wght_size_reg;
  reg                           change_state;
  reg  [DATA_IACT_BITWIDTH-1:0] mem_data_payload_reg   [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  reg  [DATA_IACT_OVERHEAD-1:0] mem_data_overhead_reg  [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  wire [DATA_IACT_BITWIDTH-1:0] storage_w              [   RAM_CELLS-1:0][IACT_WORDS_IN_RAM-1:0];
  reg  [                 4-1:0] iact_router_counter;
  reg  [                 8-1:0] kernel_y_counter;

  localparam INITIALIZE = 2'b0;
  localparam GET_PARAMETER = 2'b1;
  localparam WRITE_TO_MEMORY = 2'b10;
  localparam PARAM_EXTENDING = CALC_DATA_WIDTH - PARAM_LENGTH;

  localparam IDLE = 1'b0;
  localparam ENCODE = 1'b1;

  generate
    reg [CALC_DATA_WIDTH-1:0] fsm_enc_cycle;
    reg fsm_enc_current_state;
    reg [3:0] flat_help_var;
    integer pec, per, b;
    always @(posedge clk_i, negedge rst_ni) begin
      if (!rst_ni) begin
        fsm_enc_current_state  <= IDLE;
        iact_data_o            <= 0;
        iact_enable_o          <= 0;
        iact_choose_o          <= 0;
        fsm_enc_cycle          <= 0;
        ram_rd_addr            <= 0;
        ram_rd_en              <= 0;
        current_iact_cycle_reg <= 0;
      end else begin
        case (fsm_enc_current_state)

          IDLE: begin
            iact_data_o            <= 0;
            iact_enable_o          <= 0;
            iact_choose_o          <= ~0;
            fsm_enc_cycle          <= 0;
            ram_rd_en              <= 0;
            current_iact_cycle_reg <= 0;
            if (enable_store) begin
              ram_rd_addr <= 0;
            end
            if (enable_converter) begin
              ram_rd_en     <= 1;
              fsm_enc_cycle <= fsm_enc_cycle + 1;
            end
            if (fsm_enc_cycle >= 1) begin
              ram_rd_en <= 1;
              if (iact_ready_i != {((NUM_GLB_IACT)) {1'b1}}) begin
                fsm_enc_cycle <= fsm_enc_cycle;
              end else begin
                fsm_enc_current_state <= ENCODE;
                fsm_enc_cycle         <= 0;
              end
            end
          end

          ENCODE: begin
            fsm_enc_cycle <= fsm_enc_cycle + 1;
            ram_rd_en     <= 1;
            iact_data_o   <= ram_data_o;
            if (ram_rd_addr < address_storage) begin
              iact_enable_o <= {((NUM_GLB_IACT)) {1'b1}};
              if (fsm_enc_cycle % WORDS_PER_CYCLE == 0) begin
                ram_rd_addr <= ram_rd_addr + 1;
              end
            end
            flat_help_var = 0;
            for (pec = 0; pec < PE_X; pec = pec + 1) begin
              for (per = 0; per < PE_Y; per = per + 1) begin
                if (((pec[3:0] + per[3:0]) >=  (NUM_GLB_IACT[3:0] *  (current_iact_cycle_reg[3:0]%2))) 
                &  (pec[3:0] + per[3:0])  <  (NUM_GLB_IACT[3:0] * ((current_iact_cycle_reg[3:0]%2) + 1))
                ) begin
                  flat_help_var   = (pec[3:0] + per[3:0] - (NUM_GLB_IACT[3:0] * (current_iact_cycle_reg[3:0]%2)));
                  for (b = 0; b < $clog2(NUM_GLB_IACT + 1); b = b + 1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b] <=
                        flat_help_var[b];
                  end
                end else begin
                  for (b = 0; b < $clog2(NUM_GLB_IACT + 1); b = b + 1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b] <=
                        NUM_GLB_IACT[b];
                  end
                end
                flat_help_var = 0;
              end
            end
            // Full Iact Cycle
            if (fsm_enc_cycle[7:0] + 1 == (channels)) begin
              fsm_enc_cycle          <= 0;
              current_iact_cycle_reg <= current_iact_cycle_reg + 1;
              //All Iacts per Computing Cycle are transmitted
              if (current_iact_cycle_reg == (needed_iact_cycles_reg* wght_size_reg) - 1) begin
                fsm_enc_current_state  <= IDLE;
                ram_rd_addr            <= ram_rd_addr - (channels * (wght_size_reg - 1));
                current_iact_cycle_reg <= 0;
              end
            end
          end
          default: begin
            ram_rd_en             <= 0;
            fsm_enc_current_state <= IDLE;
          end

        endcase
      end
    end

    reg signed [7:0] y_reg;
    reg signed [7:0] x_reg             [NUM_GLB_IACT-1:0];
    /*wire       [7:0] x_test ;
    assign x_test = x_reg[0];
    wire       [7:0] b_test ;
    assign b_test = byte_var;
    wire       [7:0] r_test ;
    assign r_test = ram_var;*/
    integer          router_loop;
    reg        [1:0] fsm_current_state;
    always @(posedge clk_i, negedge rst_ni) begin
      // Reset
      if (!rst_ni) begin
        y_reg <= 0;
        for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop++) begin
          x_reg[router_loop] <= 0;
        end
      end else begin
        if ((fsm_current_state == WRITE_TO_MEMORY) | (enable_store & (fsm_current_state == GET_PARAMETER))) begin
          //localparam TESTPARAM = DATA_IACT_OVERHEAD;
          y_reg <= y - {{(8 - 4) {1'd0}}, padding_reg};
          for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop++) begin
            x_reg[router_loop] <= (iact_router_counter * NUM_GLB_IACT[7:0]) + (router_loop[7:0] + x) - {{(8 - 4){1'd0}},padding_reg};
          end
        end
        if (enable_config) begin
          for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop++) begin
            x_reg[router_loop] <= (iact_router_counter * NUM_GLB_IACT[7:0]) + (router_loop[7:0] +
            params[PARAMS_SIZE-1:3*PARAMS_SIZE/4]) - {{(8 - 4){1'd0}},padding_reg};
          end
          y_reg <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4] - {{(8 - 4) {1'd0}}, padding_reg};
        end
      end
    end

    reg [8-1:0] fsm_cycle;
    reg [8-1:0] y_cycle;
    reg [8-1:0] router_cycle;
    reg [8-1:0] addr_cycle;
    reg [4-1:0] duty_cycle;
    reg [4-1:0] duty_cycle_th;
    reg [8-1:0] ram_var;
    reg [8-1:0] byte_var;
    reg [ADDRWIDTH-1:0] ram_wr_addr_reg;
    integer r, w;
    always @(posedge clk_i, negedge rst_ni) begin
      // Reset
      if (!rst_ni) begin
        fsm_cycle              <= 0;
        y_cycle                <= 0;
        router_cycle           <= 0;
        addr_cycle             <= 0;
        fsm_current_state      <= INITIALIZE;
        x                      <= 0;
        y                      <= 0;
        channels               <= 0;
        pos                    <= 0;
        change_state           <= 0;
        ready_o                <= 0;
        ram_wr_en              <= 0;
        ram_wr_addr            <= 0;
        ram_wr_addr_reg        <= 0;
        address_storage        <= 0;
        current_cycle          <= 0;
        iact_router_counter    <= 0;
        kernel_y_counter       <= 0;
        padding_reg            <= 0;
        needed_iact_cycles_reg <= 0;
        wght_size_reg          <= 0;
        duty_cycle             <= 0;
        duty_cycle_th          <= 3;  //HERE
        x_lines_reg            <= 0;
        ram_var  = 0;
        byte_var = 0;
        for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
          for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
            mem_data_payload_reg[r][w]  <= 0;
            mem_data_overhead_reg[r][w] <= 0;
          end
        end
      end else begin
        case (fsm_current_state)
          INITIALIZE: begin
            x_lines_reg            <= 0;
            fsm_cycle              <= 0;
            y_cycle                <= 0;
            router_cycle           <= 0;
            addr_cycle             <= 0;
            fsm_current_state      <= GET_PARAMETER;
            channels               <= 0;
            x                      <= 0;
            y                      <= 0;
            pos                    <= 0;
            ready_o                <= 0;
            ram_wr_en              <= 0;
            ram_wr_addr            <= 0;
            address_storage        <= 0;
            padding_reg            <= 0;
            needed_iact_cycles_reg <= 0;
            wght_size_reg          <= 0;
            duty_cycle             <= 0;
            duty_cycle_th          <= 3;  //HERE
          end

          GET_PARAMETER: begin
            x_lines_reg         <= x_lines_i;
            fsm_cycle           <= 0;
            y_cycle             <= 1;
            router_cycle        <= 1;
            addr_cycle          <= 0;
            ram_wr_en           <= 0;
            ram_wr_addr         <= 0;
            ram_wr_addr_reg     <= 0;
            iact_router_counter <= 0;
            kernel_y_counter    <= 0;
            padding_reg         <= (wght_size_reg[2:0] - 1) / 2;
            if (enable_store) begin
              ram_wr_addr       <= address_storage;
              fsm_current_state <= WRITE_TO_MEMORY;
              duty_cycle        <= 0;
            end
          end

          WRITE_TO_MEMORY: begin
            duty_cycle <= duty_cycle + 1;
            ram_wr_addr_reg <= (iact_router_counter * wght_size_reg) + {{(ADDRWIDTH-8){1'd0}},kernel_y_counter};
            if (duty_cycle == needed_cycles_i - 1) begin
              duty_cycle <= 0;
            end
            if (duty_cycle <= duty_cycle_th) begin
              fsm_cycle    <= fsm_cycle + 1;
              y_cycle      <= y_cycle + 1;
              router_cycle <= router_cycle + 1;
              ram_wr_en    <= 0;
              if (fsm_cycle % (2 / WORDS_PER_CYCLE) == (2 / WORDS_PER_CYCLE) - 1) begin
                ram_wr_en <= 1;
              end
              if (y_cycle == ((channels * needed_iact_cycles_reg) / WORDS_PER_CYCLE) - 1) begin
                y       <= y + 1;
                y_cycle <= 0;
              end
              if (router_cycle == ((channels / WORDS_PER_CYCLE) - 1)) begin
                router_cycle        <= 0;
                iact_router_counter <= iact_router_counter + 1;
                if (iact_router_counter == needed_iact_cycles_reg - 1) begin
                  iact_router_counter <= 0;
                  kernel_y_counter    <= kernel_y_counter + 1;
                end
              end
              if (fsm_cycle % (2 / WORDS_PER_CYCLE) == 0) begin
                addr_cycle <= addr_cycle + 1;
                
                if (fsm_cycle != 0) begin
                  ram_wr_addr <= ram_wr_addr + 1;
                end else begin
                  ram_wr_addr <= address_storage;
                end
                //Reset payload to 0
                for (r = 0; r < NUM_GLB_IACT; r++) begin
                  for (w = 0; w < WORDS_PER_TRANS; w++) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end
                end
                if (fsm_cycle % (channels / WORDS_PER_CYCLE) == 0) begin
                  for (r = 0; r < NUM_GLB_IACT; r++) begin
                    for (w = 0; w < WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end else begin
                  for (r = 0; r < NUM_GLB_IACT; r++) begin
                    for (w = 0; w < WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end
              end
              for (r = 0; r < NUM_GLB_IACT; r++) begin
                for (w = 0; w < WORDS_PER_CYCLE; w++) begin
                  ram_var = (((y_reg * iact_size_xi) + x_reg[r]) / 2) % RAM_CELLS;
                  byte_var = (((x_reg[r]*channels) + ((fsm_cycle%(4/WORDS_PER_CYCLE))/(2/WORDS_PER_CYCLE))* 2)%IACT_WORDS_IN_RAM);
                  //PADDING
                  if ((
                  (0 > x_reg[r])|
                  ((iact_size_xi - 1) < x_reg[r])) | (
                  (0 > y_reg) |
                  ((iact_size_yi - 1) < y_reg)
                  )) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end else begin
                    mem_data_payload_reg[r][w] <= storage_w[ram_var[4:0]][byte_var[2:0]+w[2:0]];
                  end
                end
              end
              if (fsm_cycle == ((((needed_iact_cycles_reg * channels * (x_lines_reg))/WORDS_PER_CYCLE)))) begin
                fsm_cycle         <= 0;
                ram_wr_en         <= 0;
                current_cycle     <= current_cycle + 1;
                address_storage   <= ram_wr_addr + 1;
                fsm_current_state <= GET_PARAMETER;
              end
            end else begin
              ram_wr_en <= 0;
              if (enable_store) begin
                fsm_cycle           <= 0;
                address_storage     <= ram_wr_addr + 1;
                current_cycle       <= current_cycle + 1;
                y_cycle             <= 1;
                router_cycle        <= 1;
                addr_cycle          <= 0;
                ram_wr_en           <= 0;
                ram_wr_addr_reg     <= 0;
                iact_router_counter <= 0;
                kernel_y_counter    <= 0;
                ram_wr_addr         <= ram_wr_addr + 1;
                fsm_current_state   <= WRITE_TO_MEMORY;
                duty_cycle          <= 0;
              end
            end
          end

          default: begin
            fsm_current_state <= INITIALIZE;

          end
        endcase
        ram_var  = 0;
        byte_var = 0;
        if (enable_config) begin
          x                      <= params[PARAMS_SIZE-1:3*PARAMS_SIZE/4];
          y                      <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4];
          channels               <= params[(PARAMS_SIZE/4)-1:0];
          ready_o                <= 1;
          needed_iact_cycles_reg <= 2;
          wght_size_reg          <= 3;
        end
        if (reset_cycle_i) begin
          fsm_current_state <= INITIALIZE;
        end
      end
    end

    RAM_DP #(
        .DataWidth(WORD_BITWIDTH),
        .AddrWidth(ADDRWIDTH)
    ) iact_buffer_SP (
        .clk_i   (clk_i),
        .rd_en_i (ram_rd_en),
        .wr_en_i (ram_wr_en),
        .addr_r_i(ram_rd_addr),
        .addr_w_i(ram_wr_addr),
        .data_i  (ram_data_i),
        .data_o  (ram_data_o)
    );

    genvar r_gen, w_gen, b_gen;
    for (r_gen = 0; r_gen < NUM_GLB_IACT; r_gen = r_gen + 1) begin
      for (w_gen = 0; w_gen < WORDS_PER_TRANS; w_gen = w_gen + 1) begin
        for (b_gen = 0; b_gen < DATA_IACT_BITWIDTH; b_gen = b_gen + 1) begin
          localparam index = w_gen * IACT_DATA_DATA + r_gen * WORDS_PER_TRANS * IACT_DATA_DATA + b_gen;
          assign ram_data_i[index] = mem_data_payload_reg[r_gen][w_gen][b_gen];
        end
        for (b_gen = 0; b_gen < DATA_IACT_OVERHEAD; b_gen = b_gen + 1) begin
          localparam index = r_gen * WORDS_PER_TRANS * IACT_DATA_DATA + w_gen * IACT_DATA_DATA + DATA_IACT_BITWIDTH + b_gen;
          assign ram_data_i[index] = mem_data_overhead_reg[r_gen][w_gen][b_gen];
        end
      end
    end
    for (r_gen = 0; r_gen < RAM_CELLS; r_gen = r_gen + 1) begin
      for (w_gen = 0; w_gen < IACT_WORDS_IN_RAM; w_gen = w_gen + 1) begin
        for (b_gen = 0; b_gen < DATA_IACT_BITWIDTH; b_gen = b_gen + 1) begin
          localparam index = b_gen + w_gen * DATA_IACT_BITWIDTH;
          assign storage_w[r_gen][w_gen][b_gen] = storage_i[(DATA_IACT_BITWIDTH*IACT_WORDS_IN_RAM*r_gen)+index];
        end
      end
    end

  endgenerate

endmodule
