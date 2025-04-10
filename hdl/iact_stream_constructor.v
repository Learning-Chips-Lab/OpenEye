`timescale 1ns / 1ps

module iact_stream_constructor
#(
  // TODO wght_size changeable
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
  input                                         clk_i,
  input                                         rst_ni,
  input      [PARAMS_SIZE-1:0]                  params,
  input                                         enable_config,
  input                                         enable_store,
  input                                         enable_converter,
  input      [RAM_CELLS_WORDWIDTH-1:0]          storage_i          [RAM_CELLS-1:0],
  output reg                                    ready_o,
  input      [NUM_GLB_IACT-1:0]                 iact_ready_i,
  output reg [NUM_GLB_IACT*BITS_PER_ROUTER-1:0] iact_data_o,
  output reg [NUM_GLB_IACT-1:0]                 iact_enable_o,
  output reg [(PES*$clog2(NUM_GLB_IACT+1))-1:0] iact_choose_o
);
  reg                            ram_wr_en;
  reg  [ADDRWIDTH-1:0]           ram_wr_addr;
  wire [WORD_BITWIDTH-1:0]       ram_data_i;
  reg                            ram_rd_en;
  reg  [ADDRWIDTH-1:0]           ram_rd_addr;
  wire [WORD_BITWIDTH-1:0]       ram_data_o;
  reg  [ADDRWIDTH-1:0]           address_storage;
  reg  [PARAM_LENGTH-1:0]        x;
  reg  [PARAM_LENGTH-1:0]        y;
  reg  [PARAM_LENGTH-1:0]        iact_size;
  reg  [PARAM_LENGTH-1:0]        channels;
  reg  [PARAM_LENGTH-1:0]        current_cycle;
  reg  [15:0]                    pos;
  reg  [3:0]                     padding_reg;
  reg  [7:0]                     needed_iact_cycles_reg;
  reg  [7:0]                     current_iact_cycle_reg;
  reg  [3:0]                     wght_size_reg;
  reg                            change_state;
  reg  [DATA_IACT_BITWIDTH-1:0]  mem_data_payload_reg   [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  reg  [DATA_IACT_OVERHEAD-1:0]  mem_data_overhead_reg  [NUM_GLB_IACT-1:0][WORDS_PER_TRANS-1:0];
  wire [DATA_IACT_BITWIDTH-1:0]  storage_w              [RAM_CELLS-1:0][IACT_WORDS_IN_RAM-1:0];
  reg  [PARAM_LENGTH-1:0]        iact_router_counter;
  reg  [PARAM_LENGTH-1:0]        kernel_y_counter;
  reg  [PARAM_LENGTH-1:0]        x_var;
  reg  [PARAM_LENGTH-1:0]        y_var;
  reg  [PARAM_LENGTH-1:0]        ram_var;
  reg  [PARAM_LENGTH-1:0]        byte_var;
  reg  [32-1:0]                  flat_help_var;
  typedef enum logic [1:0] {
    INITIALIZE        = 0,
    GET_PARAMETER     = 1,
    WRITE_TO_MEMORY   = 2
  } state_t;

  typedef enum logic [0:0] {
    IDLE        = 0,
    ENCODE      = 1
  } state_c;

  generate
    reg  [31:0] fsm_enc_cycle;
    state_c fsm_enc_current_state;

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
        flat_help_var           = 0;
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
              ram_rd_addr            <= 0;
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
            for (int pec=0; pec<PE_X; pec=pec+1) begin
              for (int per=0; per<PE_Y; per=per+1) begin
                if ((32'(32'(pec) + per) >=  (NUM_GLB_IACT *  32'(current_iact_cycle_reg))) 
                &  (32'(32'(pec) + per)  <  (NUM_GLB_IACT * (32'(current_iact_cycle_reg) + 1)))
                ) begin
                  flat_help_var   = (32'(flat_help_var) + 32'(pec) + 32'(per) - 32'(NUM_GLB_IACT) * 32'(current_iact_cycle_reg));
                  for (int b=0; b<$clog2(NUM_GLB_IACT+1); b=b+1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b]
                    <= flat_help_var[b];
                  end
                end else begin
                  for (int b=0; b<$clog2(NUM_GLB_IACT+1); b=b+1) begin
                    iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+b]
                    <= NUM_GLB_IACT[b];
                  end
                end
                flat_help_var = 0;
              end
            end
            // Full Iact Cycle
            if (fsm_enc_cycle == (channels * wght_size_reg) - 1) begin
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

    reg  [31:0] fsm_cycle;
    reg  [7:0]  y_cycle;
    reg  [7:0]  router_cycle;
    reg  [7:0]  addr_cycle;
    reg  [3:0]  duty_cycle;
    reg  [3:0]  duty_cycle_th;
    reg  [3:0]  duty_cycle_reset;
    state_t fsm_current_state;

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
        x_var                   = 0;
        y_var                   = 0;
        ram_var                 = 0;
        byte_var                = 0;
        for (int r=0; r<NUM_GLB_IACT; r=r+1) begin
          for (int w=0; w<WORDS_PER_TRANS; w=w+1) begin
            mem_data_payload_reg[r][w]  <= DATA_IACT_BITWIDTH'(0);
            mem_data_overhead_reg[r][w] <= DATA_IACT_OVERHEAD'(0);
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
              if (router_cycle == 8'((32'(channels)/WORDS_PER_CYCLE) - 1)) begin
                  router_cycle        <= 0;
                  iact_router_counter <= iact_router_counter + 1;
                  if (iact_router_counter == needed_iact_cycles_reg - 1) begin
                    iact_router_counter <= 0;
                    kernel_y_counter    <= kernel_y_counter + 1;
                  end
              end
              if (fsm_cycle % (2/WORDS_PER_CYCLE) == 0) begin
                addr_cycle   <= addr_cycle + 1;
                if (addr_cycle == 8'((32'(channels)/WORDS_PER_CYCLE) - 1)) begin
                  addr_cycle  <= 0;
                  ram_wr_addr <= ram_wr_addr + 1;
                end else begin
                  ram_wr_addr <= address_storage + ADDRWIDTH'((kernel_y_counter * channels + iact_router_counter * (channels*wght_size_reg))/WORDS_PER_TRANS);
                end
                //Reset payload to 0
                for (int r=0; r<NUM_GLB_IACT; r++) begin
                  for (int w=0; w<WORDS_PER_TRANS; w++) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end
                end
                if (fsm_cycle % (32'(channels)/WORDS_PER_CYCLE) == 0) begin
                  for (int r=0; r<NUM_GLB_IACT; r++) begin
                    for (int w=0; w<WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end else begin
                  for (int r=0; r<NUM_GLB_IACT; r++) begin
                    for (int w=0; w<WORDS_PER_TRANS; w++) begin
                      mem_data_overhead_reg[r][w] <= 0;
                    end
                  end
                end
              end
              for (int r=0; r<NUM_GLB_IACT; r++) begin
                x_var = (iact_router_counter * NUM_GLB_IACT) + (8'(r) + x);
                y_var = (y);
                ram_var = ((((y_var - 8'(padding_reg))*iact_size) + (x_var-8'(padding_reg)))/2)%RAM_CELLS;
                byte_var = 8'(((((32'(x_var) - 32'(padding_reg)))*channels) + ((fsm_cycle%(4/WORDS_PER_CYCLE))/(2/WORDS_PER_CYCLE))* 2)%IACT_WORDS_IN_RAM);
                for (int w=0; w<WORDS_PER_CYCLE; w++) begin
                  //PADDING
                  if ((
                  (8'(padding_reg) > x_var)|
                  ((iact_size + 8'(padding_reg) - 1) < x_var)) | (
                  ((8'(padding_reg)) > y_var) |
                  ((iact_size + 8'(padding_reg) - 1) < y_var)
                  )) begin
                    mem_data_payload_reg[r][w] <= 0;
                  end else begin
                    mem_data_payload_reg[r][w] <= storage_w[RAM_CELLS'(ram_var)][32'(byte_var) + w];
                  end
                end

                x_var = 0;
                y_var = 0;
                ram_var = 0;
                byte_var = 0;
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
          x                      <= params[PARAMS_SIZE-1:3*PARAMS_SIZE/4];
          y                      <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4];
          iact_size              <= params[(2*PARAMS_SIZE/4)-1:PARAMS_SIZE/4];
          channels               <= params[(PARAMS_SIZE/4)-1:0];
          ready_o                <= 1;
          ram_wr_addr            <= 0;
          needed_iact_cycles_reg <= 2;
          wght_size_reg          <= 3;
        end
      end
    end

    RAM_DP #(
      .DataWidth(WORD_BITWIDTH),
      .AddrWidth(ADDRWIDTH)
    ) iact_buffer_SP (
      .clk_i    (clk_i), 
      .rd_en_i  (ram_rd_en),
      .wr_en_i  (ram_wr_en), 
      .addr_r_i (ram_rd_addr),
      .addr_w_i (ram_wr_addr),
      .data_i   (ram_data_i),
      .data_o   (ram_data_o)
    );

    genvar r, w, b;
    for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
      for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
        for (b = 0; b < DATA_IACT_BITWIDTH; b = b + 1) begin
          localparam int index = w * IACT_DATA_DATA + r * WORDS_PER_TRANS * IACT_DATA_DATA + b;
          assign ram_data_i[index] = 
                mem_data_payload_reg[r][w][b];
        end
        for (b = 0; b < DATA_IACT_OVERHEAD; b = b + 1) begin
          localparam int index = r * WORDS_PER_TRANS * IACT_DATA_DATA + w * IACT_DATA_DATA + DATA_IACT_BITWIDTH + b;
          assign ram_data_i[index] = 
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