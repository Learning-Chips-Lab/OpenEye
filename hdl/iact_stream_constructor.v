`timescale 1ns / 1ps

/// Module: iact_stream_constructor
///
/// The Input Activation Stream Constructor (iact_stream_constructor) is a sophisticated
/// module in the OpenEye architecture responsible for organizing and streaming input
/// activation data to processing elements. It handles complex data formatting,
/// buffering, and distribution patterns required for efficient neural network processing.
///
/// Key Features:
/// - Stream Organization:
///   * Configurable data formatting
///   * Multi-channel support
///   * Variable batch processing
///
/// - Memory Management:
///   * RAM-based buffering
///   * Flexible addressing
///   * Dynamic data loading
///
/// - Processing Support:
///   * PE cluster distribution
///   * Channel interleaving
///   * Padding management
///
/// - Control Mechanisms:
///   * Stream synchronization
///   * Flow control
///   * Configuration interface
///
/// Operational Modes:
/// 1. Configuration Mode:
///    - Parameter setup
///    - Stream initialization
///    - Memory organization
///
/// 2. Storage Mode:
///    - Data buffering
///    - Address management
///    - Write operations
///
/// 3. Streaming Mode:
///    - Data distribution
///    - Channel routing
///    - PE synchronization
///
/// Architecture Integration:
/// The module serves as:
/// 1. Input activation formatter
/// 2. Data distribution controller
/// 3. PE cluster interface
/// 4. Memory buffer manager
///
module iact_stream_constructor #(
    parameter  CALC_DATA_WIDTH     = 32,   // Calculation precision
    parameter  CLUSTER_COLUMNS     = 2,    // Number of PE cluster coloumns
    parameter  CLUSTER_ROWS        = 8,    // Number of PE cluster rows
    parameter  NUM_GLB_IACT        = 3,    // Global buffer interfaces
    parameter  PE_X                = 4,    // PE array width
    parameter  PE_Y                = 3,    // PE array height
    parameter  DATA_IACT_BITWIDTH  = 8,    // Activation data width
    parameter  DATA_IACT_OVERHEAD  = 4,    // Control overhead bits
    parameter  RAM_CELLS           = 32,   // Buffer depth
    parameter  RAM_CELLS_WORDWIDTH = 64,   // Buffer word width
    parameter  WORD_BITWIDTH       = 72,   // Total word width
    parameter  ADDRWIDTH           = 13,   // Address width
    localparam CLUSTERS            = CLUSTER_ROWS * CLUSTER_COLUMNS,
    localparam PES                 = PE_X * PE_Y,
    localparam IACT_WORDS_IN_RAM   = RAM_CELLS_WORDWIDTH / DATA_IACT_BITWIDTH,
    localparam IACT_DATA_DATA      = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD,
    localparam BITS_PER_ROUTER     = WORD_BITWIDTH / NUM_GLB_IACT,
    localparam WORDS_PER_TRANS     = BITS_PER_ROUTER / IACT_DATA_DATA,
    localparam IACT_CHOOSE_BITS    = $clog2(NUM_GLB_IACT+1),    
    localparam PARAMS_SIZE         = 32,
    localparam PARAM_LENGTH        = 8,
    localparam WORDS_PER_CYCLE     = 2

) (
    input                                          clk_i,
    input                                          rst_ni,
    input                                          reset_cycle_i,
    input      [                4+PARAMS_SIZE-1:0] params,
    input                                          enable_config,
    input                                          enable_store,
    input                                          enable_converter,
    input      [RAM_CELLS_WORDWIDTH*RAM_CELLS-1:0] storage_i,
    output reg                                     ready_o,
    input      [                 NUM_GLB_IACT-1:0] iact_ready_i,
    output reg [ NUM_GLB_IACT*BITS_PER_ROUTER-1:0] iact_data_o,
    output reg [                 NUM_GLB_IACT-1:0] iact_enable_o,
    output reg [ (PES*$clog2(NUM_GLB_IACT+1))-1:0] iact_choose_o,
    input      [       $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_i,
    input      [                            8-1:0] needed_iact_channel_cycles_i,
    input      [                           12-1:0] fc_size_i,
    input signed [                          8-1:0] iact_size_x_i,
    input signed [                          8-1:0] iact_size_y_i,
    input signed [                          8-1:0] iact_channels_i,
    input      [                            8-1:0] x_lines_i,
    input      [                            8-1:0] needed_wght_cycles_i,
    input      [                            4-1:0] needed_iact_router_cycles_i,
    input      [                            4-1:0] wght_size_i,
    input      [                            4-1:0] y_lines_per_calc,
    input                                          fully_connected_i,
    input      [                    ADDRWIDTH-1:0] needed_iact_buffer_words_i
);
  reg                           ram_wr_en;
  reg                           configured;
  reg  [         ADDRWIDTH-1:0] ram_wr_addr;
  reg                           ram_rd_en;
  reg  [         ADDRWIDTH-1:0] ram_rd_addr;
  wire [     WORD_BITWIDTH-1:0] ram_data_o;
  reg  [         ADDRWIDTH-1:0] address_storage;
  reg  [                 8-1:0] x;
  reg  [                 8-1:0] y;
  reg  [                 4-1:0] fsm_row_offset;
  reg  [                 8-1:0] channels;
  reg  [                 8-1:0] current_cycle;
  reg  [                 8-1:0] pos;
  reg  [                 4-1:0] padding_reg;
  reg  [                 4-1:0] needed_iact_router_cycles_reg;
  reg  [                16-1:0] current_iact_cycle_reg;
  reg  [                 8-1:0] current_iact_cycle_mod_reg;
  reg  [                 4-1:0] wght_size_reg;
  reg                           change_state;
  reg  [DATA_IACT_BITWIDTH-1:0] mem_data_payload_reg   [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  reg  [DATA_IACT_OVERHEAD-1:0] mem_data_overhead_reg  [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  wire [DATA_IACT_BITWIDTH-1:0] storage_w              [   RAM_CELLS-1:0][IACT_WORDS_IN_RAM-1:0];
  reg  [                 4-1:0] iact_router_counter;
  reg  [                 8-1:0] kernel_y_counter;
  reg  [                 8-1:0] needed_iact_transmissions;

  localparam FSM_INITIALIZE = 2'b0;
  localparam GET_PARAMETER = 2'b1;
  localparam WRITE_TO_MEMORY = 2'b10;
  localparam PARAM_EXTENDING = CALC_DATA_WIDTH - PARAM_LENGTH;

  localparam IDLE = 1'b0;
  localparam ENCODE = 1'b1;

  generate
    reg [8-1:0] fsm_enc_cycle;

    reg [  1:0] fsm_current_state;
    reg [8-1:0] ram_inc_counter;
    reg [8-1:0] ram_inc_counter_offset;
    reg         fsm_enc_current_state;
    reg [  8:0] iact_channel_counter;
    reg [ 15:0] finished_y_lines;
    reg [  7:0] iact_y_counter;
    reg [  7:0] x_line_counter;
    reg [  3:0] y_cluster_counter;
    reg [ 4-1:0] iacts_in_one_trans;

    reg [16-1:0] fsm_cycle;
    reg [ 8-1:0] x_cycle;
    reg [ 8-1:0] x_cycle_q1;
    reg [ 8-1:0] x_cycle_q2;
    reg [ 8-1:0] y_cycle_delay;
    reg [ 8-1:0] y_cycle;
    reg [12-1:0] wr_addr_1;
    reg [12-1:0] wr_addr_2;
    reg [12-1:0] wr_addr_3;
    reg [12-1:0] rd_addr_1;
    reg [12-1:0] rd_addr_2;
    reg [12-1:0] rd_addr_3;
    reg [ 8-1:0] lines_in_words;
    reg [ 8-1:0] x_pos_in_w_cycle;
    reg [ 8-1:0] router_cycle;
    reg [ 8-1:0] addr_cycle;
    reg signed [ 8-1:0] ram_var;
    reg [ 8-1:0] byte_var;
    reg [ 8-1:0] byte_var_pre_calc;
    reg [ADDRWIDTH-1:0] ram_wr_addr_reg;
    reg [7:0] transmission_amount_y_line;

    wire [         ADDRWIDTH-1:0] ram_rd_addr_reset_value;
    assign ram_rd_addr_reset_value = {{9{1'd0}},needed_iact_router_cycles_reg} * (({{5{1'd0}},iact_channels_i} + 1)/2) * (y_lines_per_calc + finished_y_lines[12:0] * y_lines_per_calc);
    integer pec, per, b;
    always @(posedge clk_i, negedge rst_ni) begin
      if (!rst_ni) begin
        fsm_enc_current_state      <= IDLE;
        iact_data_o                <= 0;
        iact_enable_o              <= 0;
        iact_choose_o              <= 0;
        fsm_enc_cycle              <= 0;
        ram_inc_counter            <= 0;
        ram_inc_counter_offset     <= 0;
        ram_rd_addr                <= 0;
        ram_rd_en                  <= 0;
        current_iact_cycle_reg     <= 0;
        current_iact_cycle_mod_reg <= 0;
        iact_channel_counter       <= 0;
        finished_y_lines           <= 0;
        iact_y_counter             <= 0;
        x_line_counter             <= 0;
        y_cluster_counter          <= 0;
        needed_iact_transmissions  <= 0;
        rd_addr_1                  <= 0;
        rd_addr_2                  <= 0;
        rd_addr_3                  <= 0;
      end else begin
        case (fsm_enc_current_state)
          IDLE: begin
            iact_data_o                <= 0;
            iact_choose_o              <= {PES{NUM_GLB_IACT[IACT_CHOOSE_BITS-1:0]}};
            fsm_enc_cycle              <= 0;
            ram_inc_counter            <= 0;
            ram_inc_counter_offset     <= 0;
            ram_rd_en                  <= 0;
            current_iact_cycle_mod_reg <= 0;
            current_iact_cycle_reg     <= 0;
            if (fsm_current_state == WRITE_TO_MEMORY) begin
              rd_addr_1                  <= (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
              rd_addr_1                  <= 0;
              rd_addr_2                  <= 0;
              if (iact_channels_i == 1) begin
                rd_addr_3 <= 0;
              end else begin
                rd_addr_3 <= 4;
              end
            end
            if (enable_store) begin
              ram_rd_addr <= 0;
            end
            if (fsm_enc_cycle == 0) begin
              iact_enable_o <= 0;
            end
            if (fsm_enc_cycle == 1) begin
              if (iact_ready_i != {((NUM_GLB_IACT)) {1'b1}}) begin
                fsm_enc_cycle <= fsm_enc_cycle;
              end else begin
                fsm_enc_current_state      <= ENCODE;
                if (iact_channels_i == 1) begin
                  if (finished_y_lines % 2 == 0) begin
                    ram_inc_counter_offset <= 1;
                  end
                  needed_iact_transmissions <= (needed_iact_router_cycles_reg * ((wght_size_reg+1)/2)) - 1;
                end else begin
                  needed_iact_transmissions <= (needed_iact_router_cycles_reg * wght_size_reg) - 1;
                end
                fsm_enc_cycle              <= 0;
                current_iact_cycle_reg     <= ~0;
                current_iact_cycle_mod_reg <= ~0;
                ram_inc_counter            <= ~0;
                if (fsm_row_offset == y_cluster_counter) begin
                  ram_rd_en <= 1;
                end
              end
            end
          end
          ENCODE: begin
            fsm_enc_cycle <= fsm_enc_cycle + 1;
            ram_rd_en     <= 0;
            iact_enable_o <= 0;
            if (fsm_row_offset == y_cluster_counter) begin
              ram_rd_en             <= 1;
              if ((current_iact_cycle_reg >> 1) != {15{1'b1}}) begin
                if ((ram_rd_addr < address_storage) | (!fully_connected_i)) begin
                  iact_enable_o <= {((NUM_GLB_IACT)){1'b1}};
                end
              end
              iact_data_o <= ram_data_o;
            end
            //Delay for one cycle
            //Check, wether amount of channels is odd
            if (((ram_inc_counter[7:0] + 1) % WORDS_PER_CYCLE[7:0] == WORDS_PER_CYCLE[7:0] - 1)) begin
              ram_rd_addr <= ram_rd_addr + 1;
              if (fsm_enc_cycle >= (needed_iact_router_cycles_reg * wght_size_reg * (iact_channels_i)) - 1) begin
                ram_rd_addr <= ram_rd_addr;
              end
            end
            for (pec = 0; pec < PE_X; pec = pec + 1) begin
              for (per = 0; per < PE_Y; per = per + 1) begin
                if (((pec[3:0] + per[3:0] + (fsm_row_offset * PE_Y)) >=  (NUM_GLB_IACT[3:0] * current_iact_cycle_mod_reg[3:0]))
                &    (pec[3:0] + per[3:0] + (fsm_row_offset * PE_Y))  <  (NUM_GLB_IACT[3:0] *(current_iact_cycle_mod_reg[3:0] + 1))
                ) begin
                  iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+: $clog2(NUM_GLB_IACT+1)] <=
                      (pec[3:0] + per[3:0]  + (fsm_row_offset * PE_Y) - (NUM_GLB_IACT[3:0] * (current_iact_cycle_mod_reg[3:0])));
                end else begin
                  iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+: $clog2(NUM_GLB_IACT+1)] <=
                      NUM_GLB_IACT;
                end
                if (fully_connected_i & (pec != 0)) begin
                  iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+: $clog2(NUM_GLB_IACT+1)] <=
                      NUM_GLB_IACT;
                end
              end
            end
            if (fsm_row_offset != y_cluster_counter) begin
              ram_rd_addr <= ram_rd_addr;
            end
            // Full Iact Cycle
            ram_inc_counter <= ram_inc_counter + 1;
            if ((ram_inc_counter[7:0] + ram_inc_counter_offset >= iact_channels_i - 1) & (ram_inc_counter[7:0] + ram_inc_counter_offset != 0)) begin
              ram_inc_counter        <= 0;
              ram_inc_counter_offset <= 0;
              if (iact_channels_i == 1) begin
                if (finished_y_lines % 2 == 0) begin
                  if (((current_iact_cycle_reg == 1))) begin
                    ram_rd_addr <= ram_rd_addr + 1;
                  end
                  if (((current_iact_cycle_reg >= 1) & (current_iact_cycle_reg <= 3)) & (finished_y_lines % 2 == 0)) begin
                    ram_inc_counter_offset <= 1;
                  end 
                end else begin
                  ram_rd_addr <= ram_rd_addr;
                  if (current_iact_cycle_reg >= 255) begin
                    ram_rd_addr            <= ram_rd_addr + 1;
                    ram_inc_counter_offset <= 1;
                  end
                end
              end
            end
            if (ram_inc_counter == 0) begin
              current_iact_cycle_reg     <= current_iact_cycle_reg + 1;
              current_iact_cycle_mod_reg <= current_iact_cycle_mod_reg + 1;
              if (current_iact_cycle_mod_reg + 1 == {{4{1'd0}},needed_iact_router_cycles_i}) begin
                current_iact_cycle_mod_reg <= 0;
              end
              //All Iacts per Computing Cycle are transmitted
              if (current_iact_cycle_reg == needed_iact_transmissions) begin
                current_iact_cycle_reg <= 0;
                //ram_rd_addr            <= (lines_in_words * 2 * 2) * (iact_channel_counter+1) + ((2 * needed_iact_router_cycles_reg) * finished_y_lines);
                ram_rd_addr            <= (1 + ram_rd_addr + (iact_size_y_i - 1) * ((iact_channels_i + 1)/2) * needed_iact_router_cycles_reg);
                iact_channel_counter   <= iact_channel_counter + 1;
                if (iact_channel_counter == needed_iact_channel_cycles_i - 1) begin
                  iact_channel_counter <= 0;
                  rd_addr_1            <= rd_addr_1 + 4;
                  ram_rd_addr          <= rd_addr_1;
                  iact_y_counter       <= iact_y_counter + 1;
                  if (iact_y_counter == needed_wght_cycles_i - 1) begin
                    iact_y_counter <= 0;
                    rd_addr_1      <= rd_addr_2 + (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
                    rd_addr_2      <= rd_addr_2 + (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
                    ram_rd_addr    <= rd_addr_2 + (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
                    x_line_counter <= x_line_counter + 1;
                    if (x_line_counter == x_lines_i - 1) begin
                      x_line_counter <= 0;
                      rd_addr_1      <= rd_addr_3 + (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
                      rd_addr_1      <= rd_addr_3;
                      rd_addr_2      <= rd_addr_3 + (lines_in_words*needed_iact_channel_cycles_i*needed_iact_router_cycles_reg*iacts_in_one_trans);
                      rd_addr_2      <= rd_addr_3;
                      if (iact_channels_i == 1) begin
                        rd_addr_3      <= (2*(1+(finished_y_lines/2)));
                      end else begin
                        rd_addr_3 <= rd_addr_3 + 4;
                      end
                      ram_rd_addr    <= rd_addr_3;
                      finished_y_lines <= finished_y_lines + 1;
                      if (finished_y_lines == iact_size_y_i - 1) begin
                        finished_y_lines <= 0;
                        ram_rd_addr      <= 0;
                        rd_addr_1        <= 0;
                        rd_addr_2        <= 0;
                        rd_addr_3        <= 0;
                      end
                    end
                  end
                end
                fsm_enc_current_state      <= IDLE;
                current_iact_cycle_mod_reg <= 0;
                ram_rd_en                  <= 0;
                ram_inc_counter            <= 0;
              end
            end
          end
          default: begin
            ram_rd_en             <= 0;
            fsm_enc_current_state <= IDLE;
          end
        endcase
        if (enable_converter & configured) begin
          fsm_enc_cycle <= 1;
        end
        if (reset_cycle_i) begin
          fsm_enc_current_state      <= IDLE;
          iact_data_o                <= 0;
          iact_enable_o              <= 0;
          iact_choose_o              <= 0;
          fsm_enc_cycle              <= 0;
          ram_inc_counter            <= 0;
          ram_rd_addr                <= 0;
          ram_rd_en                  <= 0;
          current_iact_cycle_reg     <= 0;
          current_iact_cycle_mod_reg <= 0;
          iact_channel_counter       <= 0;
          finished_y_lines           <= 0;
          iact_y_counter             <= 0;
          y_cluster_counter          <= 0;
        end
      end
    end

    reg signed [7:0] y_reg;
    reg signed [7:0] x_reg      [NUM_GLB_IACT-1:0];
    wire       [7:0] x_reg_trace;
    reg        [7:0] byte_var_trace;
    reg        [7:0] ram_var_trace;
    assign x_reg_trace = x_reg[0];
    integer          router_loop;
    always @(posedge clk_i, negedge rst_ni) begin
      // Reset
      if (!rst_ni) begin
        y_reg <= 0;
        for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop=router_loop+1) begin
          x_reg[router_loop] <= 0;
        end
      end else begin
        if ((fsm_current_state == WRITE_TO_MEMORY) | (enable_store & (fsm_current_state == GET_PARAMETER))) begin
          //localparam TESTPARAM = DATA_IACT_OVERHEAD;
          y_reg <= y - {{(8 - 4) {1'd0}}, padding_reg};
          for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop=router_loop+1) begin
            x_reg[router_loop] <= (iact_router_counter * NUM_GLB_IACT[7:0]) + (router_loop[7:0] + x) - {{(8 - 4){1'd0}},padding_reg};
          end
        end
        if (enable_config) begin
          for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop=router_loop+1) begin
            x_reg[router_loop] <= (iact_router_counter * NUM_GLB_IACT[7:0]) + (router_loop[7:0] +
            params[PARAMS_SIZE-1:3*PARAMS_SIZE/4]) - {{(8 - 4){1'd0}},padding_reg};
          end
          y_reg <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4] - {{(8 - 4) {1'd0}}, padding_reg};
        end
        if (reset_cycle_i) begin
          y_reg <= 0;
          for (router_loop = 0; router_loop < NUM_GLB_IACT; router_loop=router_loop+1) begin
            x_reg[router_loop] <= 0;
          end
        end
      end
    end

    integer signed r, w;
    always @(posedge clk_i, negedge rst_ni) begin
      // Reset
      if (!rst_ni) begin
        // Initialize the FSM cycle counter to 0 for tracking FSM state transitions
        fsm_cycle              <= 0;
        x_cycle                <= 0;
        x_cycle_q1             <= 0;
        x_cycle_q2             <= 0;
        y_cycle                <= 0;
        y_cycle_delay          <= 0;
        wr_addr_1              <= 0;
        wr_addr_2              <= 0;
        wr_addr_3              <= 0;
        iacts_in_one_trans     <= 0;
        lines_in_words         <= 0;
        router_cycle           <= 0;
        addr_cycle             <= 0;
        fsm_current_state      <= FSM_INITIALIZE;
        x                      <= 0;
        y                      <= 0;
        fsm_row_offset         <= 0;
        channels               <= 0;
        pos                    <= 0;
        change_state           <= 0;
        ready_o                <= 0;
        configured             <= 0;
        ram_wr_en              <= 0;
        ram_wr_addr            <= 0;
        ram_wr_addr_reg        <= 0;
        address_storage        <= 0;
        current_cycle          <= 0;
        iact_router_counter    <= 0;
        kernel_y_counter       <= 0;
        padding_reg            <= 0;
        needed_iact_router_cycles_reg <= 0;
        wght_size_reg          <= 0;
        x_pos_in_w_cycle       <= 0;
        ram_var                 = 0;
        transmission_amount_y_line <= 0;
        byte_var = 0;
        for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
          for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
            mem_data_payload_reg[r][w]  <= 0;
            mem_data_overhead_reg[r][w] <= 0;
          end
        end
      end else begin
        case (fsm_current_state)
          FSM_INITIALIZE: begin
            fsm_cycle              <= 0;
            x_cycle                <= 1;
            x_cycle_q1             <= 0;
            x_cycle_q2             <= 0;
            y_cycle_delay          <= 0;
            y_cycle                <= 0;
            wr_addr_1              <= 0;
            wr_addr_2              <= 0;
            wr_addr_3              <= 0;
            router_cycle           <= 0;
            addr_cycle             <= 0;
            fsm_current_state      <= GET_PARAMETER;
            x                      <= 0;
            y                      <= 0;
            fsm_row_offset         <= 0;
            channels               <= 0;
            pos                    <= 0;
            change_state           <= 0;
            ram_wr_en              <= 0;
            ram_wr_addr            <= 0;
            ram_wr_addr_reg        <= 0;
            address_storage        <= 0;
            current_cycle          <= 0;
            iact_router_counter    <= 0;
            kernel_y_counter       <= 0;
            padding_reg            <= 0;
            needed_iact_router_cycles_reg <= 0;
            wght_size_reg          <= 0;
            x_pos_in_w_cycle       <= 0;
            ready_o                <= 1;
          end

          GET_PARAMETER: begin
            fsm_cycle           <= 0;
            x_cycle             <= 1;
            x_cycle_q1          <= 0;
            x_cycle_q2          <= 0;
            y_cycle             <= 0;
            y_cycle_delay       <= 0;
            wr_addr_1           <= lines_in_words * iacts_in_one_trans * needed_iact_router_cycles_reg * needed_iact_channel_cycles_i;
            wr_addr_2           <= (iacts_in_one_trans*needed_iact_router_cycles_reg);
            wr_addr_3           <= 0;
            x_pos_in_w_cycle    <= 0;
            router_cycle        <= 0;
            addr_cycle          <= 0;
            ram_wr_en           <= 0;
            ram_wr_addr         <= 0;
            ram_wr_addr_reg     <= 0;
            iact_router_counter <= 0;
            kernel_y_counter    <= 0;
            byte_var_pre_calc   <= 0;
            padding_reg         <= (wght_size_reg[3:0] - 1) / 2;
            transmission_amount_y_line <= (padding_reg + ((iact_size_y_i+1)/2)) * needed_iact_router_cycles_reg;
            iacts_in_one_trans  <= ((iact_channels_i+1) / WORDS_PER_CYCLE);
            if (fully_connected_i) begin
              iacts_in_one_trans  <= ((iact_channels_i+2) / WORDS_PER_CYCLE);
              if ((iact_channels_i <= WORDS_PER_CYCLE)) begin
                x_cycle <= 0;
              end
            end
            lines_in_words<= (iact_size_y_i+(padding_reg*2));
            if (iact_channels_i == 1) begin
              lines_in_words <= ((iact_size_y_i+(padding_reg*2)+1)/2);
            end
            if (enable_store) begin
              ram_wr_addr       <= address_storage - 1;
              fsm_current_state <= WRITE_TO_MEMORY;
              if (0 == (iacts_in_one_trans - 1)) begin
                router_cycle        <= 0;
                iact_router_counter <= iact_router_counter + 1;
                if (iact_router_counter == needed_iact_router_cycles_reg - 1) begin
                  iact_router_counter <= 0;
                end
              end else begin
                router_cycle <= 1;
              end
            end
          end

          WRITE_TO_MEMORY: begin
            byte_var_pre_calc <= byte_var_pre_calc + 1;
            if (((byte_var_pre_calc+1) == iacts_in_one_trans)) begin
              byte_var_pre_calc <= 0;
            end
            fsm_cycle       <= fsm_cycle + 1;
            ram_wr_en       <= 0;
            if (fsm_cycle % (2 / WORDS_PER_CYCLE) == (2 / WORDS_PER_CYCLE) - 1) begin
              ram_wr_en <= 1;
            end
            router_cycle <= router_cycle + 1;
            if (router_cycle >= (iacts_in_one_trans - 1)) begin
              router_cycle        <= 0;
              iact_router_counter <= iact_router_counter + 1;
              if (iact_router_counter == needed_iact_router_cycles_reg - 1) begin
                iact_router_counter <= 0;
              end
            end
            if (fsm_cycle % (2 / WORDS_PER_CYCLE) == 0) begin
              addr_cycle <= addr_cycle + 1;
              //Reset payload to 0
              for (r = 0; r < NUM_GLB_IACT; r=r+1) begin
                for (w = 0; w < WORDS_PER_TRANS; w=w+1) begin
                  mem_data_payload_reg[r][w] <= 0;
                end
              end
              x_pos_in_w_cycle <= x_pos_in_w_cycle + 1;
              if (x_pos_in_w_cycle == (iact_channels_i / WORDS_PER_CYCLE)) begin
                x_pos_in_w_cycle <= 0;
                for (r = 0; r < NUM_GLB_IACT; r=r+1) begin
                  for (w = 0; w < WORDS_PER_TRANS; w=w+1) begin
                    mem_data_overhead_reg[r][w] <= 0;
                  end
                end
              end else begin
                for (r = 0; r < NUM_GLB_IACT; r=r+1) begin
                  for (w = 0; w < WORDS_PER_TRANS; w=w+1) begin
                    mem_data_overhead_reg[r][w] <= 0;
                  end
                end
              end
            end
            //Loops for x and y
            x_cycle <= x_cycle + 1;
            if (x_cycle == (iacts_in_one_trans * needed_iact_router_cycles_reg) - 1) begin
              x_cycle <= 0;
              y_cycle <= y_cycle + 1;
              x       <= x + (PE_X * CLUSTERS);
              if (y_cycle == x_lines_i- 1) begin
                y_cycle <= 0;
                x       <= x - ((x_lines_i - 1) * (PE_X * CLUSTERS));
                y       <= y + 1;
                if (iact_channels_i == 1) begin
                  y <= y + WORDS_PER_CYCLE;
                end
              end
            end
            //Loops for wr_address
            ram_wr_addr   <= ram_wr_addr + 1;
            x_cycle_q1 <= x_cycle;
            x_cycle_q2 <= x_cycle_q1;
            if (x_cycle_q2 == (iacts_in_one_trans * needed_iact_router_cycles_reg) - 1) begin
              wr_addr_1     <= wr_addr_1 + lines_in_words * iacts_in_one_trans * needed_iact_router_cycles_reg * needed_iact_channel_cycles_i;
              ram_wr_addr   <= wr_addr_1;
              y_cycle_delay <= y_cycle_delay + 1;
              if (y_cycle_delay == x_lines_i- 1) begin
                y_cycle_delay    <= 0;
                wr_addr_1        <= wr_addr_2 + lines_in_words * iacts_in_one_trans * needed_iact_router_cycles_reg * needed_iact_channel_cycles_i ;
                wr_addr_2        <= wr_addr_2 + (iacts_in_one_trans*needed_iact_router_cycles_reg);
                ram_wr_addr      <= wr_addr_2;
                kernel_y_counter <= kernel_y_counter + 1;
                if (kernel_y_counter == 1 - 1) begin
                  kernel_y_counter <= 0;
                end
              end
            end
            for (r = 0; r < NUM_GLB_IACT; r=r+1) begin
              for (w = 0; w < WORDS_PER_CYCLE; w=w+1) begin
                if (fully_connected_i) begin
                  byte_var = w + (((channels * PE_Y) + x_reg[r]) * iact_channels_i) + (byte_var_pre_calc/(2/WORDS_PER_CYCLE))* 2;
                  ram_var  = (byte_var  / 8);
                end else begin
                  ram_var = iact_channels_i == 1 ? w : 0;
                  ram_var = (((channels * iact_size_x_i * iact_size_y_i) + iact_channels_i * (((ram_var + y_reg) * iact_size_x_i) + x_reg[r])) / 8);
                  byte_var = (iact_size_y_i * (iact_size_x_i%2) * channels);
                  byte_var = byte_var + ((iact_size_x_i * iact_channels_i) * ((iact_channels_i == 1 ? y_reg + w: y_reg)));
                  byte_var = byte_var + (x_reg[r]*iact_channels_i);
                  byte_var = byte_var + (byte_var_pre_calc/(2/WORDS_PER_CYCLE)) * 2;
                  byte_var = (iact_channels_i == 1 ? byte_var: byte_var + w);
                end
                byte_var = byte_var % IACT_WORDS_IN_RAM;
                ram_var = ram_var % RAM_CELLS;
                if ((r == 0) & (w == 0)) begin
                  byte_var_trace = byte_var;
                  ram_var_trace = ram_var;
                end
                //PADDING
                if ((((
                (0 > x_reg[r]) |
                ((iact_size_x_i - 1) < x_reg[r])) | (
                (- w > (y_reg * iact_channels_i)) |
                ((iact_channels_i * (iact_size_y_i)) <= w + (y_reg * iact_channels_i))
                )) & !fully_connected_i) |
                (((2 * byte_var_pre_calc) + w + (((channels * NUM_GLB_IACT) + r) * iact_channels_i) >= fc_size_i) & fully_connected_i)) begin
                  mem_data_payload_reg[r][w] <= 0;
                end else begin
                  mem_data_payload_reg[r][w] <= storage_w[ram_var[4:0]][byte_var[2:0]];
                end
              end
            end
            if (fsm_cycle == (needed_iact_buffer_words_i - 1)) begin
              fsm_cycle         <= 0;
              current_cycle     <= current_cycle + 1;
              address_storage   <= ram_wr_addr + 2;

              fsm_current_state <= GET_PARAMETER;
            end
            if (enable_store) begin
              fsm_cycle           <= 0;
              address_storage     <= ram_wr_addr + 2;
              current_cycle       <= current_cycle + 1;
              if (fully_connected_i & (iact_channels_i <= WORDS_PER_CYCLE)) begin
                x_cycle             <= 0;
              end
              y_cycle             <= 0;
              router_cycle        <= 1;
              addr_cycle          <= 0;
              kernel_y_counter    <= 0;
              ram_wr_addr         <= ram_wr_addr + 1;
              fsm_current_state   <= WRITE_TO_MEMORY;
            end
          end

          default: begin
            fsm_current_state <= FSM_INITIALIZE;

          end
        endcase
        //ram_var  = 0;
        //byte_var = 0;
        if (enable_config) begin
          fsm_row_offset <= params[35:32];
          x              <= params[PARAMS_SIZE-1:3*PARAMS_SIZE/4];
          y              <= params[(3*PARAMS_SIZE/4)-1:2*PARAMS_SIZE/4];
          channels       <= params[(PARAMS_SIZE/4)-1:0];
          ready_o        <= 1;
          configured     <= 1;
        end
        needed_iact_router_cycles_reg <= needed_iact_router_cycles_i;
        wght_size_reg          <= wght_size_i;
        if (reset_cycle_i) begin
          fsm_current_state      <= FSM_INITIALIZE;
        end
      end
    end

    genvar r_gen, w_gen, b_gen;
    for (r_gen = 0; r_gen < NUM_GLB_IACT; r_gen = r_gen + 1) begin : BUFFER
      wire [BITS_PER_ROUTER-1:0]ram_data_i_w;
      wire [BITS_PER_ROUTER-1:0]ram_data_o_w;
      RAM_SP #(
          .DataWidth(BITS_PER_ROUTER),
          .AddrWidth(ADDRWIDTH),
          .Pipelined(1)
      ) iact_buffer_SP (
          .clk_i   (clk_i),
          .rd_en_i (ram_rd_en),
          .wr_en_i (ram_wr_en),
          .addr_i  (ram_rd_addr | ram_wr_addr),
          .data_i  (ram_data_i_w),
          .data_o  (ram_data_o_w)
      );
    end
    for (r_gen = 0; r_gen < NUM_GLB_IACT; r_gen = r_gen + 1) begin
      assign ram_data_o[r_gen * BITS_PER_ROUTER+:BITS_PER_ROUTER]=BUFFER[r_gen].ram_data_o_w;
      for (w_gen = 0; w_gen < WORDS_PER_TRANS; w_gen = w_gen + 1) begin
        assign BUFFER[r_gen].ram_data_i_w[w_gen * IACT_DATA_DATA +:DATA_IACT_BITWIDTH]                      = mem_data_payload_reg[r_gen][w_gen];
        assign BUFFER[r_gen].ram_data_i_w[w_gen * IACT_DATA_DATA + DATA_IACT_BITWIDTH +:DATA_IACT_OVERHEAD] = mem_data_overhead_reg[r_gen][w_gen];
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
