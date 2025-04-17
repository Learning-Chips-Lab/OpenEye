`timescale 1ns / 1ps

module iact_stream_constructor
#(
  // TODO wght_size changeable
  parameter   CALC_DATA_WIDTH     = 32,
  parameter   NUM_GLB_IACT        = 3,
  parameter   PE_X                = 4,
  parameter   PE_Y                = 3,
  parameter   DATA_IACT_BITWIDTH  = 8,
  parameter   DATA_IACT_OVERHEAD  = 4,
  parameter   RAM_CELLS           = 32,
  parameter   RAM_CELLS_WORDWIDTH = 64,
  parameter   WORD_BITWIDTH       = 72,
  parameter   ADDRWIDTH           = 1,
  localparam  PES                 = PE_X * PE_Y,
  localparam  IACT_WORDS_IN_RAM   = RAM_CELLS_WORDWIDTH/DATA_IACT_BITWIDTH,
  localparam  IACT_DATA_DATA      = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD,
  localparam  BITS_PER_ROUTER     = WORD_BITWIDTH/NUM_GLB_IACT,
  localparam  WORDS_PER_TRANS     = BITS_PER_ROUTER/IACT_DATA_DATA,
  localparam  PARAMS_SIZE         = 32,
  localparam  PARAM_LENGTH        = 8,
  localparam  WORDS_PER_CYCLE     = 2

) (
  input                                          clk_i,
  input                                          rst_ni,
  input      [PARAMS_SIZE-1:0]                   params,
  input                                          enable_config,
  input                                          enable_store,
  input                                          enable_converter,
  input      [RAM_CELLS_WORDWIDTH*RAM_CELLS-1:0] storage_i,
  output reg                                     ready_o,
  input      [NUM_GLB_IACT-1:0]                  iact_ready_i,
  output reg [NUM_GLB_IACT*BITS_PER_ROUTER-1:0]  iact_data_o,
  output reg [NUM_GLB_IACT-1:0]                  iact_enable_o,
  output reg [(PES*$clog2(NUM_GLB_IACT+1))-1:0]  iact_choose_o
);
  reg                            ram_wr_en;
  reg  [CALC_DATA_WIDTH-1:0]     ram_wr_addr;
  wire [ADDRWIDTH-1:0]           ram_wr_addr_w;
  wire [WORD_BITWIDTH-1:0]       ram_data_i;
  reg                            ram_rd_en;
  reg  [CALC_DATA_WIDTH-1:0]     ram_rd_addr;
  wire [ADDRWIDTH-1:0]           ram_rd_addr_w;
  wire [WORD_BITWIDTH-1:0]       ram_data_o;
  reg  [CALC_DATA_WIDTH-1:0]     address_storage;
  reg  [CALC_DATA_WIDTH-1:0]     x;
  reg  [CALC_DATA_WIDTH-1:0]     y;
  reg  [CALC_DATA_WIDTH-1:0]     iact_size;
  reg  [CALC_DATA_WIDTH-1:0]     channels;
  reg  [CALC_DATA_WIDTH-1:0]     current_cycle;
  reg  [CALC_DATA_WIDTH-1:0]     pos;
  reg  [CALC_DATA_WIDTH-1:0]     padding_reg;
  reg  [CALC_DATA_WIDTH-1:0]     needed_iact_cycles_reg;
  reg  [CALC_DATA_WIDTH-1:0]     current_iact_cycle_reg;
  reg  [CALC_DATA_WIDTH-1:0]     wght_size_reg;
  reg                            change_state;
  reg  [DATA_IACT_BITWIDTH-1:0]  mem_data_payload_reg   [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  reg  [DATA_IACT_OVERHEAD-1:0]  mem_data_overhead_reg  [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  wire [DATA_IACT_BITWIDTH-1:0]  storage_w              [RAM_CELLS-1:0][IACT_WORDS_IN_RAM-1:0];
  reg  [CALC_DATA_WIDTH-1:0]     iact_router_counter;
  reg  [CALC_DATA_WIDTH-1:0]     kernel_y_counter;

  localparam INITIALIZE      = 2'b0;
  localparam GET_PARAMETER   = 2'b1;
  localparam WRITE_TO_MEMORY = 2'b10;
  localparam PARAM_EXTENDING = CALC_DATA_WIDTH - PARAM_LENGTH;

  localparam IDLE = 1'b0;
  localparam ENCODE = 1'b1;
  assign ram_rd_addr_w = ram_rd_addr[ADDRWIDTH-1:0];
  assign ram_wr_addr_w = ram_wr_addr[ADDRWIDTH-1:0];

  generate
    reg  [CALC_DATA_WIDTH-1:0] fsm_enc_cycle;
    reg fsm_enc_current_state;

    integer pec,per,flat_help_var,b;
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

          IDLE : begin
            iact_data_o            <= 0;
            iact_enable_o          <= 0;
            iact_choose_o          <= ~0;
            fsm_enc_cycle          <= 0;
            ram_rd_en              <= 0;
            current_iact_cycle_reg <= 0;
            if (enable_store) begin
              ram_rd_addr          <= 0;
            end
            if (enable_converter) begin
              ram_rd_en     <= 1;
              fsm_enc_cycle <= fsm_enc_cycle + 1;
            end
            if (fsm_enc_cycle >= 1) begin
              ram_rd_en             <= 1;
              if (iact_ready_i !={((NUM_GLB_IACT)){1'b1}}) begin
                fsm_enc_cycle <= fsm_enc_cycle;
              end else begin
                fsm_enc_current_state <= ENCODE;
                fsm_enc_cycle         <= 0;
              end
            end
          end

          ENCODE : begin
            fsm_enc_cycle         <= fsm_enc_cycle + 1;
            ram_rd_en             <= 1;
            iact_data_o           <= ram_data_o;
            if (ram_rd_addr < address_storage) begin
              iact_enable_o         <= {((NUM_GLB_IACT)){1'b1}};
              if (fsm_enc_cycle % WORDS_PER_CYCLE == 0) begin
                  ram_rd_addr <= ram_rd_addr + 1;
              end
            end
            flat_help_var = 0;
            for (pec=0; pec<PE_X; pec=pec+1) begin
              for (per=0; per<PE_Y; per=per+1) begin
                if (((pec + per) >=  (NUM_GLB_IACT *  current_iact_cycle_reg)) 
                &  (pec + per)  <  (NUM_GLB_IACT * (current_iact_cycle_reg + 1'b1))
                ) begin
                  flat_help_var   = (flat_help_var + pec + per - (NUM_GLB_IACT * current_iact_cycle_reg));
                  for (b=0; b<$clog2(NUM_GLB_IACT+1); b=b+1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b]
                    <= flat_help_var[b];
                  end
                end else begin
                  for (b=0; b<$clog2(NUM_GLB_IACT+1); b=b+1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b]
                    <= NUM_GLB_IACT[b];
                  end
                end
                flat_help_var = 0;
              end
            end
            // Full Iact Cycle
            if (fsm_enc_cycle + 1 == (channels * wght_size_reg)) begin
              fsm_enc_cycle          <= 0;
              current_iact_cycle_reg <= current_iact_cycle_reg + 1;
              //All Iacts per Computing Cycle are transmitted
              if (current_iact_cycle_reg == needed_iact_cycles_reg - 1) begin
                fsm_enc_current_state  <= IDLE;
              end
            end
          end
          default : begin
            ram_rd_en             <= 0;
            fsm_enc_current_state <= IDLE;
          end

        endcase
      end
    end

    reg  [CALC_DATA_WIDTH-1:0]   fsm_cycle;
    reg  [CALC_DATA_WIDTH-1:0]   y_cycle;
    reg  [CALC_DATA_WIDTH-1:0]   router_cycle;
    reg  [CALC_DATA_WIDTH-1:0]   addr_cycle;
    reg  [CALC_DATA_WIDTH-1:0]   duty_cycle;
    reg  [CALC_DATA_WIDTH-1:0]   duty_cycle_th;
    reg  [CALC_DATA_WIDTH-1:0]   duty_cycle_reset;
    reg  [1:0]                   fsm_current_state;
    integer x_var;
    integer y_var;
    integer ram_var;
    integer byte_var;
    integer temp_var;
    integer r,w;
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
        iact_size              <= 0;
        channels               <= 0;
        pos                    <= 0;
        change_state           <= 0;
        ready_o                <= 0;
        ram_wr_en              <= 0;
        ram_wr_addr            <= 0;
        address_storage        <= 0;
        current_cycle          <= 0;
        iact_router_counter    <= 0;
        kernel_y_counter       <= 0;
        padding_reg            <= 0;
        needed_iact_cycles_reg <= 0;
        wght_size_reg          <= 0;
        duty_cycle             <= 0;
        duty_cycle_th          <= 3; //HERE
        duty_cycle_reset       <= 3; //HERE
        for (r=0; r<NUM_GLB_IACT; r=r+1) begin
          for (w=0; w<WORDS_PER_TRANS; w=w+1) begin
            mem_data_payload_reg[r][w]  <= 0;
            mem_data_overhead_reg[r][w] <= 0;
          end
        end
      end else begin
        case (fsm_current_state)
          INITIALIZE : begin
            fsm_cycle              <= 0;
            y_cycle                <= 0;
            router_cycle           <= 0;
            addr_cycle             <= 0;
            fsm_current_state      <= GET_PARAMETER;
            iact_size              <= 0;
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
            duty_cycle_th          <= 3; //HERE
            duty_cycle_reset       <= 3; //HERE
          end

          GET_PARAMETER : begin
            fsm_cycle           <= 0;
            y_cycle             <= 0;
            router_cycle        <= 0;
            addr_cycle          <= 0;
            ram_wr_en           <= 0;
            ram_wr_addr         <= 0;
            iact_router_counter <= 0;
            kernel_y_counter    <= 0;
            if (enable_store) begin
              ram_wr_addr       <= address_storage;
              padding_reg       <= (wght_size_reg-1)/2;
              fsm_current_state <= WRITE_TO_MEMORY;
              duty_cycle        <= 0;
            end
          end

          WRITE_TO_MEMORY : begin
            duty_cycle <= duty_cycle + 1;
            if (duty_cycle == duty_cycle_reset) begin
              duty_cycle <= 0;
            end
            if (duty_cycle <= duty_cycle_th) begin
              fsm_cycle    <= fsm_cycle + 1;
              y_cycle      <= y_cycle + 1;
              router_cycle <= router_cycle + 1;
              ram_wr_en  <= 0;
              if (fsm_cycle % (2/WORDS_PER_CYCLE) == (2/WORDS_PER_CYCLE) - 1) begin
                ram_wr_en  <= 1;
              end
              if (y_cycle == ((channels * needed_iact_cycles_reg)/WORDS_PER_CYCLE) - 1) begin
                y       <= y + 1;
                y_cycle <= 0;
              end
              if (router_cycle == ((channels/WORDS_PER_CYCLE) - 1)) begin
                  router_cycle        <= 0;
                  iact_router_counter <= iact_router_counter + 1;
                  if (iact_router_counter == needed_iact_cycles_reg - 1) begin
                    iact_router_counter <= 0;
                    kernel_y_counter    <= kernel_y_counter + 1;
                  end
              end
              if (fsm_cycle % (2/WORDS_PER_CYCLE) == 0) begin
                addr_cycle   <= addr_cycle + 1;
                if (addr_cycle == ((channels/WORDS_PER_CYCLE) - 1)) begin
                  addr_cycle  <= 0;
                  ram_wr_addr <= ram_wr_addr + 1;
                end else begin
                  temp_var = iact_router_counter * wght_size_reg;
                  temp_var = temp_var + kernel_y_counter;
                  temp_var = temp_var * channels;
                  temp_var = (temp_var/WORDS_PER_TRANS);
                  temp_var = temp_var + address_storage;
                  ram_wr_addr <= temp_var;
                end
                //Reset payload to 0
                for (r=0; r<NUM_GLB_IACT; r++) begin
                  for (w=0; w<WORDS_PER_TRANS; w++) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end
                end
                if (fsm_cycle % (channels/WORDS_PER_CYCLE) == 0) begin
                  for (r=0; r<NUM_GLB_IACT; r++) begin
                    for (w=0; w<WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end else begin
                  for (r=0; r<NUM_GLB_IACT; r++) begin
                    for (w=0; w<WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end
              end
              for (r=0; r<NUM_GLB_IACT; r++) begin
                for (w=0; w<WORDS_PER_CYCLE; w++) begin
                  x_var = (iact_router_counter * NUM_GLB_IACT) + (r + x);
                  y_var = (y);
                  ram_var = ((((y_var - padding_reg)*iact_size) + (x_var-padding_reg))/2)%RAM_CELLS;
                  byte_var = (((((x_var - padding_reg))*channels) + ((fsm_cycle%(4/WORDS_PER_CYCLE))/(2/WORDS_PER_CYCLE))* 2)%IACT_WORDS_IN_RAM);
                  //PADDING
                  if ((
                  (padding_reg > x_var)|
                  ((iact_size + padding_reg - 1) < x_var)) | (
                  ((padding_reg) > y_var) |
                  ((iact_size + padding_reg - 1) < y_var)
                  )) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end else begin
                    mem_data_payload_reg[r][w] <= storage_w[ram_var][byte_var + w];
                  end
                end
              end
              if (fsm_cycle == (((needed_iact_cycles_reg * channels * wght_size_reg)/WORDS_PER_CYCLE))) begin
                fsm_cycle         <= 0;
                ram_wr_en         <= 0;
                current_cycle     <= current_cycle + 1;
                address_storage   <= ram_wr_addr + 1;
                fsm_current_state <= GET_PARAMETER;
              end
            end else begin
              ram_wr_en  <= 0;
            end
          end

          default : begin
            fsm_current_state      <= INITIALIZE;

          end
        endcase
        if (enable_config == 1) begin
          x                      <= {{PARAM_EXTENDING{1'd0}},{params[PARAMS_SIZE-1:3*PARAMS_SIZE/4]}};
          y                      <= {{PARAM_EXTENDING{1'd0}},{params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4]}};
          iact_size              <= {{PARAM_EXTENDING{1'd0}},{params[(2*PARAMS_SIZE/4)-1:PARAMS_SIZE/4]}};
          channels               <= {{PARAM_EXTENDING{1'd0}},{params[(PARAMS_SIZE/4)-1:0]}};
          ready_o                <= 1;
          ram_wr_addr            <= 0;
          needed_iact_cycles_reg <= 2;
          wght_size_reg          <= 3;
        end
      end
      temp_var = 0;
    end

    RAM_DP #(
      .DataWidth(WORD_BITWIDTH),
      .AddrWidth(ADDRWIDTH)
    ) iact_buffer_SP (
      .clk_i    (clk_i), 
      .rd_en_i  (ram_rd_en),
      .wr_en_i  (ram_wr_en), 
      .addr_r_i (ram_rd_addr_w),
      .addr_w_i (ram_wr_addr_w),
      .data_i   (ram_data_i),
      .data_o   (ram_data_o)
    );

    genvar r_gen, w_gen, b_gen;
    for (r_gen = 0; r_gen < NUM_GLB_IACT; r_gen = r_gen + 1) begin
      for (w_gen = 0; w_gen < WORDS_PER_TRANS; w_gen = w_gen + 1) begin
        for (b_gen = 0; b_gen < DATA_IACT_BITWIDTH; b_gen = b_gen + 1) begin
          localparam index = w_gen * IACT_DATA_DATA + r_gen * WORDS_PER_TRANS * IACT_DATA_DATA + b_gen;
          assign ram_data_i[index] = 
                mem_data_payload_reg[r_gen][w_gen][b_gen];
        end
        for (b_gen = 0; b_gen < DATA_IACT_OVERHEAD; b_gen = b_gen + 1) begin
          localparam index = r_gen * WORDS_PER_TRANS * IACT_DATA_DATA + w_gen * IACT_DATA_DATA + DATA_IACT_BITWIDTH + b_gen;
          assign ram_data_i[index] = 
                mem_data_overhead_reg[r_gen][w_gen][b_gen];
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