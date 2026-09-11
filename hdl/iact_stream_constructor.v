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
    parameter  SPARSITY_EN         = 1,    // Enabled Sparsity Overhead
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
    localparam IACT_DATA_DATA      = DATA_IACT_BITWIDTH + (DATA_IACT_OVERHEAD * SPARSITY_EN),
    localparam BITS_PER_ROUTER     = WORD_BITWIDTH / NUM_GLB_IACT,
    localparam WORDS_PER_TRANS     = BITS_PER_ROUTER / IACT_DATA_DATA,
    localparam IACT_CHOOSE_BITS    = $clog2(NUM_GLB_IACT+1),    
    localparam PARAMS_SIZE         = 32,
    localparam PARAM_LENGTH        = 8,
    localparam WORDS_PER_CYCLE     = 2

) (
    input                                                clk_i,
    input                                                rst_ni,
    input                                                reset_cycle_i,
    input      [                      6+PARAMS_SIZE-1:0] params,
    input                                                enable_config,
    input                                                enable_store,
    input                                                enable_converter,
    input      [(2*DATA_IACT_BITWIDTH*NUM_GLB_IACT)-1:0] storage_i,
    output reg                                           ready_o,
    input      [                       NUM_GLB_IACT-1:0] iact_ready_i,
    output reg [       NUM_GLB_IACT*BITS_PER_ROUTER-1:0] iact_data_o,
    output reg [                       NUM_GLB_IACT-1:0] iact_enable_o,
    output reg [       (PES*$clog2(NUM_GLB_IACT+1))-1:0] iact_choose_o,
    input      [             $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_i,
    input      [                                  8-1:0] needed_iact_channel_cycles_i,
    input      [                                 14-1:0] fc_size_i,
    input signed [                               12-1:0] iact_x_add_up,
    input signed [                                8-1:0] iact_size_y_i,
    input signed [                                4-1:0] iact_channels_per_pe_i,
    input      [                                    1:0] channel_div_trans,
    input      [                                  8-1:0] x_lines_i,
    input      [                                  8-1:0] needed_wght_cycles_i,
    input      [                                  4-1:0] needed_iact_router_cycles_i,
    input      [                                  6-1:0] wght_size_x_i,
    input      [                                  4-1:0] wght_size_y_i,
    input      [                                  3-1:0] stride_x_i,
    input      [                                  3-1:0] stride_y_i,
    input      [                                    3:0] iact_x_per_cluster_i,
    input      [                                  4-1:0] y_lines_per_calc,
    input                                                fully_connected_i,
    input      [                                   11:0] needed_iact_buffer_words_i,
    input      [                                  8-1:0] iact_words_per_compute,
    input      [                                    7:0] rd_loop_limit_0,
    input      [                                    7:0] rd_loop_limit_1,
    input      [                                    7:0] rd_loop_limit_2,
    input      [                                    7:0] rd_loop_limit_3,
    input      [                                    7:0] rd_loop_limit_4,
    input      [                                   11:0] rd_addr_inc_0,
    input      [                                   11:0] rd_addr_inc_1,
    input      [                                   11:0] rd_addr_inc_2,
    input      [                                   11:0] rd_addr_inc_3,
    input      [                                   11:0] rd_addr_inc_4,
    input      [                                   11:0] wr_loop_limit_0,
    input      [                                    7:0] wr_loop_limit_1,
    input      [                                    7:0] wr_loop_limit_2,
    input      [                                   11:0] wr_addr_inc_0,
    input      [                                   11:0] wr_addr_inc_1,
    input      [                                   11:0] wr_addr_inc_2,
    input      [                                    3:0] padding_x,
    input      [                                    3:0] padding_y
);
  reg                           ram_wr_en;
  reg                           ram_wr_en_q;
  reg                           configured;
  reg  [         ADDRWIDTH-1:0] ram_wr_addr;
  reg  [         ADDRWIDTH-1:0] ram_wr_addr_q;
  reg                           ram_rd_en;
  reg  [         ADDRWIDTH-1:0] ram_rd_addr;
  wire [     WORD_BITWIDTH-1:0] ram_data_o;
  reg  [         ADDRWIDTH-1:0] address_storage;
  reg  [                 4-1:0] fsm_row_offset;
  reg  [                 8-1:0] x_start;
  reg  [                12-1:0] x_range_lower_bound;
  reg  [                 8-1:0] channels;
  reg  [                 8-1:0] channels_q;
  reg  [                 8-1:0] pos;
  reg  [                 4-1:0] needed_iact_router_cycles_reg;
  reg  [                16-1:0] current_iact_cycle_reg;
  reg  [                 8-1:0] current_iact_cycle_mod_reg;
  reg  [                 6-1:0] wght_size_x;
  reg  [                 4-1:0] wght_size_y;
  reg                           change_state;
  reg  [DATA_IACT_BITWIDTH-1:0] mem_data_payload_reg   [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  reg  [DATA_IACT_OVERHEAD-1:0] mem_data_overhead_reg  [NUM_GLB_IACT-1:0][  WORDS_PER_TRANS-1:0];
  reg  [                 4-1:0] iact_router_counter;
  reg  [                 8-1:0] kernel_y_counter;

  localparam FSM_INITIALIZE = 2'b0;
  localparam GET_PARAMETER = 2'b1;
  localparam WRITE_TO_MEMORY = 2'b10;
  localparam PARAM_EXTENDING = CALC_DATA_WIDTH - PARAM_LENGTH;

  localparam IDLE = 1'b0;
  localparam ENCODE = 1'b1;

  generate
    reg [12-1:0] fsm_enc_cycle;

    reg [  1:0] fsm_current_state;
    reg [8-1:0] ram_inc_counter;
    reg         ram_inc_counter_offset;
    reg         fsm_enc_current_state;
    reg [  8:0] iact_channel_counter;
    reg [ 15:0] finished_y_lines;
    reg [  7:0] wght_counter;
    reg [  7:0] x_line_counter;
    reg [ 4-1:0] iacts_in_one_trans;

    reg [16-1:0] fsm_cycle;
    reg [12-1:0] wr_addr_0;
    reg [12-1:0] wr_addr_1;
    reg [12-1:0] wr_addr_2;
    reg [12-1:0] rd_addr_0;
    reg [12-1:0] rd_addr_1;
    reg [12-1:0] rd_addr_2;
    reg [12-1:0] rd_addr_3;
    reg [12-1:0] rd_addr_4;
    reg [7:0] wr_cycle_loop_cnt_0;
    reg [7:0] wr_cycle_loop_cnt_1;
    reg [7:0] wr_cycle_loop_cnt_2;
    reg [7:0] rd_cycle_loop_cnt_0;
    reg [7:0] rd_cycle_loop_cnt_1;
    reg [7:0] rd_cycle_loop_cnt_2;
    reg [7:0] rd_cycle_loop_cnt_3;
    reg [7:0] rd_cycle_loop_cnt_4;
    reg [ 8-1:0] x_pos_in_w_cycle;
    reg [ 8-1:0] router_cycle;
  wire [12-1:0] rd_addr_inc_0_next;
  wire [12-1:0] rd_addr_inc_1_next;
  wire [12-1:0] rd_addr_inc_2_next;
  wire [12-1:0] rd_addr_inc_3_next;
  wire [12-1:0] rd_addr_inc_4_next;
  assign rd_addr_inc_0_next = rd_addr_0 + rd_addr_inc_0;
  assign rd_addr_inc_1_next = rd_addr_1 + rd_addr_inc_1;
  assign rd_addr_inc_2_next = rd_addr_2 + rd_addr_inc_2;
  assign rd_addr_inc_3_next = rd_addr_3 + rd_addr_inc_3;
  assign rd_addr_inc_4_next = rd_addr_4 + rd_addr_inc_4;
  wire [3:0]values_per_word;
  assign values_per_word = fully_connected_i ? 1 : iact_channels_per_pe_i;

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
        wght_counter               <= 0;
        x_line_counter             <= 0;
        rd_addr_0                  <= 0;
        rd_addr_1                  <= 0;
        rd_addr_2                  <= 0;
        rd_addr_3                  <= 0;
        rd_addr_4                  <= 0;
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
            if (enable_store) begin
              ram_rd_addr <= 0;
              rd_addr_0   <= 0;
              rd_addr_1   <= 0;
              rd_addr_2   <= 0;
              rd_addr_3   <= 0;
              rd_addr_4   <= 0;
            end
            if (fsm_enc_cycle == 0) begin
              iact_enable_o <= 0;
            end
            if (fsm_enc_cycle == 1) begin
              if (iact_ready_i != {((NUM_GLB_IACT)) {1'b1}}) begin
                fsm_enc_cycle <= fsm_enc_cycle;
              end else begin
                fsm_enc_current_state      <= ENCODE;
                fsm_enc_cycle              <= 0;
                current_iact_cycle_reg     <= ~0;
                // The window register is pre-decremented so its first advance
                // lands on row 0. It used to start at ~0 with a +1 step; since
                // 4cc9e07 it steps by NUM_GLB_IACT, and ~0 + NUM_GLB_IACT wraps
                // to NUM_GLB_IACT-1, which skipped rows 0..NUM_GLB_IACT-2 and
                // never met the wrap compare, so iact_choose selected almost no
                // PE and conv layers stalled without iacts.
                current_iact_cycle_mod_reg <= 8'd0 - NUM_GLB_IACT;
                ram_inc_counter            <= ~0;
                if (fsm_row_offset == 0) begin
                  ram_rd_en <= 1;
                end
              end
            end
          end
          ENCODE: begin
            fsm_enc_cycle <= fsm_enc_cycle + 1;
            ram_rd_en     <= 0;
            iact_enable_o <= 0;
            if (fsm_row_offset == 0) begin
              ram_rd_en             <= 1;
              if ((current_iact_cycle_reg >> 1) != {15{1'b1}}) begin
                if ((ram_rd_addr <= address_storage)) begin
                  iact_enable_o <= {((NUM_GLB_IACT)){1'b1}};
                end
              end
              iact_data_o <= ram_data_o;
            end
            //Delay for one cycle
            //Check, wether amount of channels is odd
            for (pec = 0; pec < PE_X; pec = pec + 1) begin
              for (per = 0; per < PE_Y; per = per + 1) begin
                if ((((pec[3:0] * stride_x_i) + per[3:0] + fsm_row_offset) >=  (current_iact_cycle_mod_reg))
                &    ((pec[3:0] * stride_x_i) + per[3:0] + fsm_row_offset)  <  (current_iact_cycle_mod_reg[3:0] + NUM_GLB_IACT)
                ) begin
                  iact_choose_o[per*PE_X*$clog2(NUM_GLB_IACT+1)+pec*$clog2(NUM_GLB_IACT+1)+: $clog2(NUM_GLB_IACT+1)] <=
                      ((pec[3:0] * stride_x_i) + per[3:0]  + fsm_row_offset - (current_iact_cycle_mod_reg[3:0]));
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
            // Full Iact Cycle
            ram_inc_counter <= ram_inc_counter + 1;
            if ((ram_inc_counter[7:0] + ram_inc_counter_offset >= values_per_word - 1) & (ram_inc_counter[7:0] + ram_inc_counter_offset != 0)) begin
              ram_inc_counter        <= 0;
              ram_inc_counter_offset <= 0;
            end
            if (((ram_inc_counter[7:0] + 1) % WORDS_PER_CYCLE[7:0] == WORDS_PER_CYCLE[7:0] - 1)) begin
              if (ram_inc_counter == 0) begin
                  current_iact_cycle_mod_reg <= current_iact_cycle_mod_reg + NUM_GLB_IACT;
                  if (current_iact_cycle_mod_reg == (needed_iact_router_cycles_i - 1) * NUM_GLB_IACT) begin
                    current_iact_cycle_mod_reg <= 0;
                  end
              end
              //All Iacts per Computing Cycle are transmitted
              current_iact_cycle_reg <= 0;
              rd_addr_0              <= rd_addr_inc_0_next;
              ram_rd_addr            <= rd_addr_inc_0_next;
              rd_cycle_loop_cnt_0    <= rd_cycle_loop_cnt_0 + 1;
              if (rd_cycle_loop_cnt_0 == rd_loop_limit_0) begin
                rd_cycle_loop_cnt_0  <= 0;
                rd_addr_0            <= rd_addr_inc_1_next;
                rd_addr_1            <= rd_addr_inc_1_next;
                ram_rd_addr          <= rd_addr_inc_1_next;
              end
            end
            if (fsm_enc_cycle == iact_words_per_compute) begin
              ram_rd_addr         <= rd_addr_inc_2_next;
              rd_addr_0           <= rd_addr_inc_2_next;
              rd_addr_1           <= rd_addr_inc_2_next;
              rd_addr_2           <= rd_addr_inc_2_next;
              rd_cycle_loop_cnt_1 <= rd_cycle_loop_cnt_1 + 1;
              if (rd_cycle_loop_cnt_1 == rd_loop_limit_1) begin
                rd_cycle_loop_cnt_1 <= 0;
                ram_rd_addr         <= rd_addr_3;
                rd_addr_0           <= rd_addr_3;
                rd_addr_1           <= rd_addr_3;
                rd_addr_2           <= rd_addr_3;
                rd_cycle_loop_cnt_2 <= rd_cycle_loop_cnt_2 + 1;
                if (rd_cycle_loop_cnt_2 == rd_loop_limit_2) begin
                  rd_cycle_loop_cnt_2 <= 0;
                  ram_rd_addr         <= rd_addr_inc_3_next;
                  rd_addr_0           <= rd_addr_inc_3_next;
                  rd_addr_1           <= rd_addr_inc_3_next;
                  rd_addr_2           <= rd_addr_inc_3_next;
                  rd_addr_3           <= rd_addr_inc_3_next;
                  rd_cycle_loop_cnt_3 <= rd_cycle_loop_cnt_3 + 1;
                  if (rd_cycle_loop_cnt_3 == rd_loop_limit_3) begin
                    rd_cycle_loop_cnt_3 <= 0;
                    ram_rd_addr         <= rd_addr_inc_4_next;
                    rd_addr_0           <= rd_addr_inc_4_next;
                    rd_addr_1           <= rd_addr_inc_4_next;
                    rd_addr_2           <= rd_addr_inc_4_next;
                    rd_addr_3           <= rd_addr_inc_4_next;
                    rd_addr_4           <= rd_addr_inc_4_next;
                  end
                end
              end
              fsm_enc_current_state      <= IDLE;
              rd_cycle_loop_cnt_0        <= 0;
              current_iact_cycle_mod_reg <= 0;
              ram_rd_en                  <= 0;
              ram_inc_counter            <= 0;
              fsm_enc_cycle              <= 0;
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
          wght_counter               <= 0;
        end
      end
    end
reg enable_write_to_storage;
    integer signed r, w;
    always @(posedge clk_i, negedge rst_ni) begin
      // Reset
      if (!rst_ni) begin
        // Initialize the FSM cycle counter to 0 for tracking FSM state transitions
        fsm_cycle                     <= 0;
        enable_write_to_storage                    <= 0;
        wr_addr_0                     <= 0;
        wr_addr_1                     <= 0;
        wr_addr_2                     <= 0;
        iacts_in_one_trans            <= 0;
        router_cycle                  <= 0;
        fsm_current_state             <= FSM_INITIALIZE;
        fsm_row_offset                <= 0;
        x_range_lower_bound           <= 0;
        x_start                       <= 0;
        channels                      <= 0;
        channels_q                    <= 0;
        pos                           <= 0;
        change_state                  <= 0;
        ready_o                       <= 0;
        configured                    <= 0;
        ram_wr_en                     <= 0;
        ram_wr_en_q                   <= 0;
        ram_wr_addr                   <= 0;
        ram_wr_addr_q                 <= 0;
        address_storage               <= 0;
        iact_router_counter           <= 0;
        kernel_y_counter              <= 0;
        needed_iact_router_cycles_reg <= 0;
        wght_size_x                   <= 0;
        wght_size_y                   <= 0;
        x_pos_in_w_cycle              <= 0;
        wr_cycle_loop_cnt_0           <= 0;
        wr_cycle_loop_cnt_1           <= 0;
        wr_cycle_loop_cnt_2           <= 0;
        rd_cycle_loop_cnt_0           <= 0;
        rd_cycle_loop_cnt_1           <= 0;
        rd_cycle_loop_cnt_2           <= 0;
        rd_cycle_loop_cnt_3           <= 0;
        rd_cycle_loop_cnt_4           <= 0;
        for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
          for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
            mem_data_payload_reg[r][w]  <= 0;
            mem_data_overhead_reg[r][w] <= 0;
          end
        end
      end else begin
        channels_q          <= channels;
        ram_wr_addr_q       <= ram_wr_addr;
        case (fsm_current_state)
          FSM_INITIALIZE: begin
            fsm_cycle              <= 0;
            enable_write_to_storage               <= 0;
            wr_addr_1              <= 0;
            wr_addr_2              <= 0;
            router_cycle           <= 0;
            fsm_current_state      <= GET_PARAMETER;
            fsm_row_offset         <= 0;
            channels               <= 0;
            pos                    <= 0;
            change_state           <= 0;
            ram_wr_en              <= 0;
            ram_wr_en_q            <= 0;
            ram_wr_addr            <= 0;
            address_storage        <= -1;
            iact_router_counter    <= 0;
            kernel_y_counter       <= 0;
            needed_iact_router_cycles_reg <= 0;
            wght_size_x                   <= 0;
            wght_size_y                   <= 0;
            x_pos_in_w_cycle              <= 0;
            ready_o                       <= 1;
          end
          GET_PARAMETER: begin
            fsm_cycle           <= 0;
            if (enable_config) begin
              wr_addr_1      <= wr_addr_inc_1;
              wr_addr_2      <= wr_addr_inc_2;
            end
            ram_wr_addr                <= 0;
            x_pos_in_w_cycle           <= 0;
            router_cycle               <= 0;
            ram_wr_en                  <= 0;
            ram_wr_en_q                <= 0;
            iact_router_counter        <= 0;
            kernel_y_counter           <= 0;
            iacts_in_one_trans         <= ((values_per_word+1) / WORDS_PER_CYCLE);
            if (enable_store) begin
              ram_wr_addr         <= address_storage;
              fsm_current_state   <= WRITE_TO_MEMORY;
              wr_addr_0           <= 0;
              wr_addr_1           <= 0;
              wr_addr_2           <= 0;
              wr_cycle_loop_cnt_0 <= ~0;
              x_pos_in_w_cycle    <= 0;
              x_range_lower_bound <= x_start;
              
              if (0 == (iacts_in_one_trans - 1)) begin
                router_cycle        <= 0;
                iact_router_counter <= iact_router_counter + 1;
                if (iact_router_counter == needed_iact_router_cycles_reg - 1) begin
                  iact_router_counter <= 0;
                end
              end
              router_cycle <= 0;
            end
          end

          WRITE_TO_MEMORY: begin
            ram_wr_en_q <= ram_wr_en;
            if (ram_wr_en) begin
              for (r = 0; r < NUM_GLB_IACT; r = r + 1) begin
                for (w = 0; w < WORDS_PER_TRANS; w = w + 1) begin
                  mem_data_payload_reg[r][w] <= storage_i[w*8+:8];
                end
              end
            end
            if (CLUSTERS != 1) begin
              router_cycle <= router_cycle + 1;
              if (router_cycle == (2 - 1)) begin
                router_cycle <= 0;
              end
              if (router_cycle == 0) begin
                enable_write_to_storage <= 0;
                if ((x_pos_in_w_cycle >= x_range_lower_bound) & (x_pos_in_w_cycle < x_range_lower_bound + PE_X + PE_Y - 1)) begin
                  enable_write_to_storage <= 1;
                end
                x_pos_in_w_cycle <= x_pos_in_w_cycle + 1;
                if (x_pos_in_w_cycle == iact_x_add_up + 2 - 1) begin
                  x_pos_in_w_cycle <= 0;
                end
              end
              if (ram_wr_en & !enable_write_to_storage) begin
                x_range_lower_bound <= x_start;
                if (iact_x_add_up > x_range_lower_bound + (CLUSTERS * PE_X)) begin
                  x_range_lower_bound <= x_range_lower_bound + (CLUSTERS * PE_X);
                end
              end
            end else begin
              enable_write_to_storage <= 1;
            end
            ram_wr_en <= 0;
            if (enable_write_to_storage) begin
              fsm_cycle   <= fsm_cycle + 1;
              ram_wr_en   <= 1;
              //Loops for wr_address
              ram_wr_addr         <= wr_addr_0;
              wr_addr_0           <= wr_addr_0 + wr_addr_inc_0;
              wr_cycle_loop_cnt_0 <= wr_cycle_loop_cnt_0 + 1;
              if (wr_cycle_loop_cnt_0 == wr_loop_limit_0) begin
                wr_cycle_loop_cnt_0 <= 0;
                wr_addr_0           <= wr_addr_1 + wr_addr_inc_0;
                wr_addr_1           <= wr_addr_1 + wr_addr_inc_1;
                ram_wr_addr         <= wr_addr_1;
                wr_cycle_loop_cnt_1 <= wr_cycle_loop_cnt_1 + 1;
                if (wr_cycle_loop_cnt_1 == wr_loop_limit_1) begin
                  wr_cycle_loop_cnt_1 <= 0;
                  wr_addr_0           <= wr_addr_2 + wr_addr_inc_0;
                  wr_addr_1           <= wr_addr_2 + wr_addr_inc_1;
                  wr_addr_2           <= wr_addr_2 + wr_addr_inc_2;
                  ram_wr_addr         <= wr_addr_2;
                  kernel_y_counter    <= 0;
                end
              end
              if (fsm_cycle == needed_iact_buffer_words_i) begin
                fsm_cycle         <= 0;
                fsm_current_state <= GET_PARAMETER;
                ram_wr_en         <= 0;
              end
            end
          end

          default: begin
            fsm_current_state <= FSM_INITIALIZE;

          end
        endcase
        if (enable_config) begin
          fsm_row_offset <= params[35:32];
          channels       <= params[(PARAMS_SIZE/4)-1:0];
          x_start        <= params[(3*PARAMS_SIZE/4)+:8];
          ready_o        <= 1;
          configured     <= 1;
        end
        needed_iact_router_cycles_reg <= needed_iact_router_cycles_i;
        wght_size_x                   <= wght_size_x_i;
        wght_size_y                   <= wght_size_y_i;
        if (reset_cycle_i) begin
          fsm_current_state <= FSM_INITIALIZE;
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
          .wr_en_i (ram_wr_en_q),
          .addr_i  (ram_rd_addr | ram_wr_addr_q),
          .data_i  (ram_data_i_w),
          .data_o  (ram_data_o_w)
      );
    end
    for (r_gen = 0; r_gen < NUM_GLB_IACT; r_gen = r_gen + 1) begin
      assign ram_data_o[r_gen * BITS_PER_ROUTER+:BITS_PER_ROUTER]=BUFFER[r_gen].ram_data_o_w;
      for (w_gen = 0; w_gen < WORDS_PER_TRANS; w_gen = w_gen + 1) begin
        assign BUFFER[r_gen].ram_data_i_w[w_gen * IACT_DATA_DATA +:DATA_IACT_BITWIDTH]                      = mem_data_payload_reg[r_gen][w_gen];
        if (SPARSITY_EN == 1) begin
          assign BUFFER[r_gen].ram_data_i_w[w_gen * IACT_DATA_DATA + DATA_IACT_BITWIDTH +:DATA_IACT_OVERHEAD] = mem_data_overhead_reg[r_gen][w_gen];
        end
      end
    end
  endgenerate
endmodule
