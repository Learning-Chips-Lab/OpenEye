// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// =============================================================================
// Module: data_pipeline_wght
// =============================================================================
// Purpose:
//   Implements the Weight data pipeline within a single PE of the OpenEye
//   sparse systolic array.  It receives packed weight words from the global
//   weight bus, decomposes them into individual weight values together with
//   their overhead (position) tags, and distributes them across two scratchpad
//   (SPAD) memories with distinct roles:
//
//     First SPAD  (wght address / filter-count SPAD):
//       Each entry records how many non-zero weight values have been stored
//       within the current filter group so far.  The MAC engine reads this to
//       know when to advance to the next filter.  Address advances once per
//       filter boundary (controlled by first_spad_max_i / filters_w).
//
//     Second SPAD (wght data SPAD):
//       Stores the actual weight bytes alongside a corrected 4-bit overhead
//       tag that encodes the within-filter position of each value.
//       Word format:
//         [SECOND_SPAD_DATA-1 : SECOND_PAYLOAD_WIDTH+SECOND_OVERHEAD_WIDTH] -- upper bits from data_i (pass-through)
//         [SECOND_PAYLOAD_WIDTH+SECOND_OVERHEAD_WIDTH-1 : SECOND_PAYLOAD_WIDTH] -- overhead_output (4-bit corrected tag)
//         [SECOND_PAYLOAD_WIDTH-1 : 0]                                        -- weight byte
//
// Sparsity encoding (SPARSITY_EN = 1, default):
//   The module reconstructs a corrected overhead tag (overhead_output) that
//   wraps around the filter boundary:  when the accumulated overhead in
//   data_storage_2 would exceed filters_w, the tag is adjusted by subtracting
//   filters_w so the MAC engine sees positions relative to the start of the
//   current filter.  In dense mode (SPARSITY_EN = 0) the second SPAD word is
//   stored verbatim (data_storage_2 unchanged).
//
// Two-cycle pipeline:
//   The write to both SPADs is delayed by one cycle relative to the data_i
//   capture, implementing a two-stage pipeline:
//     Cycle N  : data_i captured into data_storage_2; overhead/address
//                accumulators updated; first_spad_data_delay incremented.
//     Cycle N+1: enable_delay fires; data_storage_2 is written to second SPAD
//                via premade_spad_2_output; first_spad_data_delay written to
//                first SPAD via first_spad_data_o.
//   This one-cycle latency is required because overhead_output is a
//   combinational function of data_storage_2, which is itself registered from
//   data_i on the same cycle.
//
// Filter-boundary tracking:
//   The overhead accumulator (overhead_reg) counts the total number of
//   non-zero weight positions seen so far within the current filter.  Each
//   incoming DATA_WIDTH word contributes overhead_w non-zero positions
//   (sum of the 4-bit overhead fields of its two sub-words).  When
//   overhead_reg + overhead_next_word >= filters_w (the filter size),
//   the first SPAD address advances and overhead_reg wraps by subtracting
//   filters_w.  The over_ending flag handles the edge case where the
//   wrap overshoots by exactly 1.
//
// Clock / Reset:
//   Single clock domain (clk_i).  Asynchronous active-low reset (rst_ni).
//   compute_i is a synchronous per-PE clear that resets address, overhead,
//   and data state.  compute_delay (one-cycle delayed compute_i) clears the
//   storage and address registers one cycle later to handle the pipeline flush.
//
// =============================================================================

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------
//
//  DATA_WIDTH                [default 24]
//    Width of the input data bus (data_i).  Must equal 2 * SECOND_SPAD_DATA
//    because the module always processes exactly two SECOND_SPAD_DATA-wide
//    sub-words per input cycle.
//
//  FIRST_SPAD_ADDR           [default 16]
//    Depth (number of entries) of the first SPAD (filter-count SPAD).
//
//  FIRST_SPAD_DATA           [default 4]
//    Width in bits of each first SPAD entry.  Holds the running non-zero
//    count per filter group.
//
//  SECOND_SPAD_ADDR          [default 16]
//    Depth (number of entries) of the second SPAD (weight data SPAD).
//
//  SECOND_SPAD_DATA          [default 12]
//    Full width in bits of each second SPAD entry:
//      = SECOND_OVERHEAD_WIDTH (4-bit tag) + SECOND_PAYLOAD_WIDTH (8-bit weight).
//    Note: the current implementation always assumes DATA_WIDTH / SECOND_SPAD_DATA = 2
//    (two sub-words per input word).
//
//  SECOND_PAYLOAD_WIDTH      [default 8]
//    Width of the actual weight byte stored in the second SPAD.
//
//  SPARSITY_EN               [default 1]
//    When 1: sparse mode — overhead tag in second SPAD output is corrected
//            for filter-boundary wrapping via overhead_output.
//    When 0: dense mode — second SPAD output equals data_storage_2 verbatim.
//
//  FIRST_SPAD_ADDR_BITWIDTH  [= $clog2(FIRST_SPAD_ADDR)]
//    Address bit-width for the first SPAD output port.
//
//  SECOND_SPAD_ADDR_BITWIDTH [= $clog2(SECOND_SPAD_ADDR)]
//    Address bit-width for the second SPAD output port.
//
//  FIRST_SPAD_DATA_CYCLE     [= DATA_WIDTH / FIRST_SPAD_DATA]
//    Number of first-SPAD-granularity sub-words in one input word.
//    Used only to size cycle_counter.
//
//  SECOND_SPAD_DATA_CYCLE    [= DATA_WIDTH / SECOND_SPAD_DATA]
//    Number of second-SPAD-granularity sub-words in one input word (= 2).
//
//  SECOND_OVERHEAD_WIDTH     [= SECOND_SPAD_DATA - SECOND_PAYLOAD_WIDTH]
//    Width of the overhead tag field packed into each second SPAD entry
//    (= 4 bits in the default configuration).

// -----------------------------------------------------------------------------
// Port Descriptions
// -----------------------------------------------------------------------------
//
//  clk_i              - System clock.
//  rst_ni             - Asynchronous active-low reset.  Clears all state.
//  compute_i          - Synchronous per-computation clear pulse.  Resets SPAD
//                       addresses, overhead accumulators, data buffers.
//                       compute_delay (internal) fires one cycle later to
//                       flush the pipeline registers.
//
//  data_i[DATA_WIDTH-1:0]
//                     - Packed input word containing two consecutive
//                       SECOND_SPAD_DATA-wide weight sub-words.
//                       Sub-word 0: data_i[11:0]  (input_words_w[0])
//                       Sub-word 1: data_i[23:12] (input_words_w[1])
//                       Each sub-word format: { overhead_tag[3:0], weight_byte[7:0] }
//
//  enable_i           - When 1: module captures data_i and updates internal
//                       accumulators.  The write to both SPADs occurs on the
//                       following cycle (enable_delay path).
//                       When 0: outputs are idle; accumulators hold state.
//
//  first_spad_words_o[$clog2(FIRST_SPAD_ADDR+1)-1:0]
//                     - Total number of valid entries written to the first SPAD.
//                       Updated as first_spad_addr_o + 2 each time the first
//                       SPAD address advances past a filter boundary.
//
//  first_spad_max_i[$clog2(FIRST_SPAD_ADDR+1)-1:0]
//                     - Filter size limit: number of non-zero weight positions
//                       per filter group.  When 0, interpreted as 16 (filters_w).
//                       Controls when overhead_reg wraps and first_spad_addr_o steps.
//
//  second_spad_words_o[$clog2(SECOND_SPAD_ADDR+1)-1:0]
//                     - Count of valid entries written to the second SPAD.
//                       Updated as second_spad_addr_o + 2 on each delayed write.
//                       Cleared on the first enable_i cycle after compute_i
//                       (via compute_sent flag).
//
//  first_spad_addr_o[FIRST_SPAD_ADDR_BITWIDTH-1:0]
//                     - Write address into the first SPAD.  Advances on the
//                       delayed write cycle when overhead_new_calc_reg >= filters_w
//                       (a filter boundary was crossed).
//
//  first_spad_data_o[FIRST_SPAD_DATA-1:0]
//                     - Data written to the first SPAD on each delayed write:
//                       the value of first_spad_data_delay (running non-zero
//                       count, incremented once per non-zero data_i word).
//
//  first_spad_en_o    - Write-enable for the first SPAD.  Asserted on the
//                       cycle after a non-zero data_i was seen (enable_delay
//                       && data_storage_2 != 0).
//
//  second_spad_addr_o[SECOND_SPAD_ADDR_BITWIDTH-1:0]
//                     - Write address into the second SPAD.  Advances by 1
//                       on each delayed write cycle when second_spad_data_o != 0.
//                       Resets to 0 when enable_delay is low.
//
//  second_spad_data_o[SECOND_SPAD_DATA-1:0]
//                     - Data written to the second SPAD: premade_spad_2_output,
//                       which is a corrected version of data_storage_2 with
//                       the overhead tag replaced by overhead_output.
//                       In dense mode: equals data_storage_2 verbatim.
//
//  second_spad_en_o   - Write-enable for the second SPAD.  Asserted on the
//                       delayed write cycle (enable_delay && data_storage_2 != 0).

module data_pipeline_wght #(
    parameter DATA_WIDTH                = 24,
    parameter FIRST_SPAD_ADDR           = 16,
    parameter FIRST_SPAD_DATA           = 4,
    parameter SECOND_SPAD_ADDR          = 16,
    parameter SECOND_SPAD_DATA          = 12,
    parameter SECOND_PAYLOAD_WIDTH      = 8,
    parameter SPARSITY_EN               = 1,  // 1=sparse mode (default), 0=dense mode
    parameter FIRST_SPAD_ADDR_BITWIDTH  = $clog2(FIRST_SPAD_ADDR),
    parameter SECOND_SPAD_ADDR_BITWIDTH = $clog2(SECOND_SPAD_ADDR),
    parameter FIRST_SPAD_DATA_CYCLE     = DATA_WIDTH / FIRST_SPAD_DATA,
    parameter SECOND_OVERHEAD_WIDTH     = SECOND_SPAD_DATA - SECOND_PAYLOAD_WIDTH
) (
    input clk_i,
    input rst_ni,
    input compute_i,

    input      [                DATA_WIDTH-1 : 0] data_i,
    input                                         enable_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [  FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output reg [           FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                    first_spad_en_o,

    output reg [ SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output reg [          SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                    second_spad_en_o
);

  // ---------------------------------------------------------------------------
  // Internal Registers
  // ---------------------------------------------------------------------------

  // enable_delay: one-cycle delayed copy of enable_i.
  //   The actual SPAD writes are triggered on the cycle when enable_delay is
  //   high (i.e. one cycle after enable_i), because second SPAD data depends
  //   on data_storage_2 which is registered from data_i on the same cycle.
  reg                                             enable_delay;

  // first_spad_addr_delay: shadow of first_spad_addr_o, advanced one cycle
  //   ahead of the output port.  Incremented in the enable_i path when a
  //   filter boundary is crossed; propagated to first_spad_addr_o in the
  //   compute_delay path.
  reg         [   FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_delay;

  // first_spad_data_delay: running count of non-zero weight words written
  //   to the second SPAD since the last filter boundary.  Incremented in
  //   the enable_i (capture) path; written to first_spad_data_o in the
  //   enable_delay (write) path.
  reg         [            FIRST_SPAD_DATA-1 : 0] first_spad_data_delay;

  // data_storage_1 (signed): reference level for the next-channel-counter
  //   calculation.  Holds first_spad_data_delay at the time of the last
  //   filter-boundary crossing.  Used in next_channel_counter to detect
  //   when enough non-zero values have been accumulated for a new filter.
  //   Initialised to -filters_w when enable_i is low (no data arriving).
  reg  signed [            FIRST_SPAD_DATA-1 : 0] data_storage_1;

  // data_storage_2: one-cycle pipeline register capturing data_i each cycle
  //   (regardless of enable_i).  Used as the source for the delayed second
  //   SPAD write via premade_spad_2_output.
  reg  signed [           SECOND_SPAD_DATA-1 : 0] data_storage_2;

  // temp_acc_overhead: accumulated overhead count since the last
  //   filter-boundary crossing.  Incremented by overhead_w each enable_i
  //   cycle; cleared to 0 when next_channel_counter crosses 0.
  //   Used together with overhead_reg in the next_channel_counter formula.
  reg         [            FIRST_SPAD_DATA-1 : 0] temp_acc_overhead;

  // address_temp_2: second SPAD write-address accumulator (one cycle ahead).
  //   Incremented every enable_i cycle (one per non-zero data_i word).
  //   Written to second_spad_addr_o in the enable_delay write path.
  //   Wraps to 0 at SECOND_SPAD_ADDR - 1.
  reg         [     $clog2(SECOND_SPAD_ADDR) : 0] address_temp_2;

  // cycle_counter: sub-word cycle counter within one DATA_WIDTH input word.
  //   Incremented each enable_i cycle.  Currently wraps to 0 at cycle 0
  //   (effectively always 0 since only one word is processed per enable
  //   cycle in the default configuration).  Kept for potential future use
  //   with wider DATA_WIDTH values.
  reg         [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;

  // overhead_reg: running count of accumulated non-zero overhead positions
  //   within the current filter group.  Incremented by (2 + overhead_w)
  //   each enable_i cycle (2 for the two sub-words in this word, plus the
  //   overhead tags from both sub-words indicating additional non-zero entries).
  //   When overhead_reg + overhead_next_word >= filters_w, it wraps by
  //   subtracting filters_w (filter-boundary crossing detected).
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)  :0] overhead_reg;

  // overhead_delay_reg: one-cycle delayed copy of overhead_reg (before the
  //   current cycle's update).  Used in the combinational overhead_output
  //   computation to reconstruct the correct tag for data_storage_2.
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)  :0] overhead_delay_reg;

  // overhead_new_calc_reg: snapshot of overhead_reg taken at the start of
  //   the current enable_i cycle (before incrementing).  Used in the
  //   enable_delay path to detect whether a filter boundary was crossed
  //   during the previous cycle (overhead_new_calc_reg >= filters_w).
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)  :0] overhead_new_calc_reg;

  // compute_delay: one-cycle delayed copy of compute_i.
  //   Triggers a second pipeline-flush step one cycle after compute_i:
  //   clears data_storage_1/2, first_spad_addr registers.
  reg                                             compute_delay;

  // compute_sent: flag set by compute_i pulse.
  //   On the first enable_i == 1 cycle after compute_i, clears
  //   second_spad_words_o (deferred clear to avoid race with write path).
  reg                                             compute_sent;

  // over_ending: single-cycle flag asserted when overhead_reg wraps and
  //   overshoots by exactly 1 (overhead_reg == filters_w + 1).
  //   Adjusts the wrap to subtract an extra 1 from overhead_reg and also
  //   subtracts 1 from overhead_output so the position tag stays correct.
  reg                                             over_ending;

  // ---------------------------------------------------------------------------
  // Internal Wires
  // ---------------------------------------------------------------------------

  // input_words_w[0..1]: the two SECOND_SPAD_DATA-wide sub-words extracted
  //   from data_i.
  //   input_words_w[0] = data_i[11:0]   (first sub-word)
  //   input_words_w[1] = data_i[23:12]  (second sub-word)
  //   Each sub-word: { overhead_tag[3:0], weight_byte[7:0] }
  wire        [             SECOND_SPAD_DATA-1:0] input_words_w [0:2-1];

  // overhead_w: total overhead contribution from both sub-words in the
  //   current data_i word.
  //   = input_words_w[0][11:8] + input_words_w[1][11:8]
  //   (sum of both 4-bit overhead tags).
  wire        [                              3:0] overhead_w;

  // next_channel_counter (signed): heuristic counter that fires when enough
  //   non-zero values have been accumulated for the current filter.
  //   Formula:  first_spad_data_o + ((temp_acc_overhead + overhead_w) / 2)
  //             - data_storage_1 - filters_w
  //   When >= 0: a filter boundary has been reached; first_spad_addr steps
  //   and data_storage_1 / temp_acc_overhead are reset.
  wire signed [                              7:0] next_channel_counter;

  // overhead_pos: byte-position counter within the current filter group,
  //   incremented by 2 each enable_i cycle (2 sub-words per DATA_WIDTH word).
  //   Reset to 0 when next_channel_counter crosses 0.
  //   Used in missingvalue to detect whether the current word straddles
  //   a filter boundary.
  reg         [                              7:0] overhead_pos;

  // filters_w: effective filter size (first_spad_max_i, defaulting to 16
  //   when first_spad_max_i == 0).
  wire        [  $clog2(FIRST_SPAD_ADDR+1)-1 : 0] filters_w;

  // premade_spad_2_output: the corrected second SPAD word assembled for output.
  //   Sparse mode:  { data_storage_2[23:12], overhead_output[3:0], data_storage_2[7:0] }
  //     (replaces the overhead tag field with the wrapped/corrected overhead_output).
  //   Dense mode:   data_storage_2 verbatim.
  wire        [           SECOND_SPAD_DATA-1 : 0] premade_spad_2_output;

  // overhead_output: corrected 4-bit overhead tag for the word currently in
  //   data_storage_2.  Wraps relative to the filter boundary:
  //   If (data_storage_2[11:8] != 0) AND (overhead_delay_reg + data_storage_2[11:8] >= filters_w):
  //     overhead_output = data_storage_2[11:8] + overhead_delay_reg - filters_w - over_ending
  //   Otherwise:
  //     overhead_output = data_storage_2[11:8]  (no wrap, pass through unchanged).
  wire        [                            3 : 0] overhead_output;

  // overhead_next_word: the overhead tag of the first sub-word of the
  //   current data_i, used to look ahead and detect an imminent filter
  //   boundary before updating overhead_reg.
  wire        [                            3 : 0] overhead_next_word;

  // ---------------------------------------------------------------------------
  // Combinational Assignments
  // ---------------------------------------------------------------------------

  // filters_w: treat first_spad_max_i == 0 as 16 (full filter depth).
  assign filters_w = first_spad_max_i == 0 ? 16 : first_spad_max_i;

  // next_channel_counter: signed comparison to detect filter-boundary crossing.
  //   Positive (>= 0) means the accumulated count has reached filters_w,
  //   so the first SPAD address should step and counters reset.
  assign next_channel_counter = first_spad_data_o + ((temp_acc_overhead + overhead_w)/2) - data_storage_1 - filters_w;

  // Unpack data_i into two 12-bit sub-words.
  genvar w_gen;
  for (w_gen = 0; w_gen < 2; w_gen = w_gen + 1) begin
    assign input_words_w[w_gen] = data_i[(12*w_gen)+:12];
  end

  // overhead_w: total overhead (non-zero position count) contributed by
  //   both sub-words in the current data_i word.
  assign overhead_w = input_words_w[0][SECOND_PAYLOAD_WIDTH+:SECOND_OVERHEAD_WIDTH]
                    + input_words_w[1][SECOND_PAYLOAD_WIDTH+:SECOND_OVERHEAD_WIDTH];

  // overhead_next_word: look-ahead overhead tag from sub-word 0.
  //   Used to detect a filter boundary before updating overhead_reg.
  assign overhead_next_word = input_words_w[0][11:8];

  // premade_spad_2_output: corrected second SPAD output word.
  //   Sparse: replace bits [11:8] (overhead tag) with overhead_output.
  //   Dense:  pass data_storage_2 unchanged.
  if (SPARSITY_EN) begin : gen_sparse
    assign premade_spad_2_output = {data_storage_2[23:12], overhead_output, data_storage_2[7:0]};
  end else begin : gen_dense
    assign premade_spad_2_output = data_storage_2;
  end

  // overhead_output: corrected 4-bit overhead tag for data_storage_2.
  //   Condition: tag is non-zero AND accumulated overhead crosses filters_w.
  //   If so: subtract filters_w (and 1 extra if over_ending) to wrap the tag
  //   into the next filter's address space.
  //   Otherwise: pass the stored tag through unchanged.
  assign overhead_output = (data_storage_2[11:8] != 0) & overhead_delay_reg + data_storage_2[11:8] >= filters_w
                         ? data_storage_2[11:8] + overhead_delay_reg - filters_w - over_ending
                         : data_storage_2[11:8];

  // ---------------------------------------------------------------------------
  // Main Sequential Process
  // ---------------------------------------------------------------------------
  // Single always block handling reset, compute clear, and normal operation.
  //
  // Execution order within the else branch (all non-blocking):
  //   1. Defaults: de-assert SPAD enables, update enable_delay, capture data_i.
  //   2. Delayed write path (enable_delay && data_storage_2 != 0):
  //      Writes premade_spad_2_output to the second SPAD and first_spad_data_delay
  //      to the first SPAD; advances addresses; updates word counts.
  //   3. Immediate capture path (enable_i && data_i != 0):
  //      Updates overhead accumulators, cycle/address counters, and
  //      first_spad_data_delay.  Detects filter boundaries via next_channel_counter.
  //   4. Idle path (enable_i == 0 || data_i == 0):
  //      Resets data_storage_1 and overhead_pos to the negative-filter-size
  //      sentinel so the next data burst starts cleanly.
  //   5. compute_i path: immediate address/accumulator reset + compute_sent flag.
  //   6. compute_delay path: one-cycle-later pipeline flush.
  // ---------------------------------------------------------------------------
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      // -----------------------------------------------------------------------
      // Asynchronous Reset — clear all outputs and internal state.
      // -----------------------------------------------------------------------
      enable_delay          <= 0;
      first_spad_words_o    <= 0;
      second_spad_words_o   <= 0;
      first_spad_addr_o     <= 0;
      first_spad_addr_delay <= 0;
      first_spad_data_o     <= 0;
      first_spad_data_delay <= 0;
      first_spad_en_o       <= 0;
      second_spad_addr_o    <= 0;
      second_spad_en_o      <= 0;
      data_storage_1        <= 0;
      data_storage_2        <= 0;
      temp_acc_overhead     <= 0;
      address_temp_2        <= 0;
      cycle_counter         <= 0;
      second_spad_data_o    <= 0;
      overhead_reg          <= 0;
      overhead_delay_reg    <= 0;
      overhead_new_calc_reg <= 0;
      compute_delay         <= 0;
      compute_sent          <= 0;
      overhead_pos          <= 0;
      over_ending           <= 0;
    end else begin

      // -----------------------------------------------------------------------
      // Defaults: one-cycle pulse outputs de-asserted every cycle.
      // data_storage_2 always captures data_i (pipeline register).
      // over_ending cleared every cycle; set in the enable_i path if needed.
      // -----------------------------------------------------------------------
      first_spad_en_o   <= 0;
      first_spad_data_o <= 0;
      enable_delay      <= enable_i;
      first_spad_en_o   <= 0;
      second_spad_en_o  <= 0;
      data_storage_2    <= data_i;   // pipeline register: always captures input
      over_ending       <= 0;

      // -----------------------------------------------------------------------
      // Delayed write path (pipeline stage 2):
      // Fires one cycle after a non-zero data_i was captured.
      // Writes both SPADs and advances their addresses/word counts.
      //
      // Conditions: enable_delay == 1 AND data_storage_2 != 0
      //   (the captured word has at least one non-zero field).
      //
      // Actions:
      //   - Assert first_spad_en_o and second_spad_en_o.
      //   - Write premade_spad_2_output to second_spad_data_o.
      //   - Filter-boundary check: if overhead_new_calc_reg >= filters_w
      //     (boundary was crossed in the previous cycle), advance
      //     first_spad_addr_o by 1 and update first_spad_words_o.
      //   - Advance second_spad_addr_o if the current second_spad_data_o is
      //     non-zero (sparse guard — skip if the output word is zero).
      //   - Write first_spad_data_delay into first_spad_data_o.
      //   - Update second_spad_words_o = second_spad_addr_o + 2 (count
      //     includes the word being written now).
      //   - When enable_delay is low: reset second_spad_addr_o to 0.
      // -----------------------------------------------------------------------
      if (enable_delay & (data_storage_2 != 0)) begin
        first_spad_en_o       <= 1;
        second_spad_en_o      <= 1;
        second_spad_data_o    <= premade_spad_2_output;
        if (overhead_new_calc_reg >= filters_w) begin
          // Filter boundary crossed in previous cycle: step first SPAD address.
          first_spad_addr_o  <= first_spad_addr_o + 1;
          first_spad_words_o <= first_spad_addr_o + 2;
        end
        if (second_spad_data_o != 0) begin
          // Non-zero second SPAD word: advance address (sparse: skip zeros).
          second_spad_addr_o    <= second_spad_addr_o + 1;
        end
        first_spad_data_o     <= first_spad_data_delay;
        second_spad_words_o   <= second_spad_addr_o + 2;
      end else begin
        // enable_delay low: reset second SPAD write address.
        second_spad_addr_o <= 0;
      end

      // -----------------------------------------------------------------------
      // Immediate capture path (pipeline stage 1):
      // Fires on every enable_i cycle where data_i is non-zero.
      // Updates accumulators and detects filter boundaries.
      //
      // Actions:
      //   - Clear compute_sent (active data arriving; deferred clear done).
      //   - Latch overhead_delay_reg = overhead_reg (snapshot before update,
      //     used by overhead_output in the next cycle).
      //   - If compute_sent was set: clear second_spad_words_o (deferred
      //     clear after the compute_i flush).
      //   - Increment cycle_counter (wraps if cycle_counter == 0, i.e. always
      //     stays at 0 in the default single-word-per-cycle configuration).
      //   - Increment address_temp_2 (wrap at SECOND_SPAD_ADDR - 1).
      //   - Increment first_spad_data_delay (non-zero word count for first SPAD).
      //   - Update overhead_reg by adding (2 + overhead_w):
      //       2 for the two sub-word slots in this word,
      //       overhead_w for the non-zero positions indicated by the tags.
      //     Filter-boundary wrap: if overhead_reg + overhead_next_word >= filters_w,
      //       subtract filters_w from the new overhead_reg.
      //       over_ending: if after subtracting we are still one too high
      //         (overhead_reg == filters_w + 1), subtract 1 more and set over_ending.
      //   - overhead_new_calc_reg: snapshot of overhead_reg before update,
      //     used in the enable_delay path to detect the boundary.
      //       Special case: if overhead_reg < filters_w, update to
      //         overhead_reg + 2 + overhead_next_word (look-ahead).
      //   - Accumulate temp_acc_overhead += overhead_w.
      //   - Increment overhead_pos by 2 (byte offset within filter).
      //
      //   next_channel_counter check (filter boundary):
      //     If next_channel_counter >= 0:
      //       - Save data_storage_1 = first_spad_data_delay.
      //       - Reset overhead_pos to 0.
      //       - If data_storage_1 >= 0: reset temp_acc_overhead to 0 and
      //         advance first_spad_addr_delay.
      //
      // Idle case (enable_i == 0 || data_i == 0):
      //   Reset data_storage_1 to -filters_w (sentinel so next burst starts
      //   at the right offset) and overhead_pos to filters_w.
      // -----------------------------------------------------------------------
      if (enable_i == 1 & (data_i != 0)) begin
        compute_sent       <= 0;
        overhead_delay_reg <= overhead_reg;  // snapshot before this cycle's update
        if (compute_sent) begin
          // Deferred second_spad_words_o clear after compute_i.
          second_spad_words_o <= 0;
        end
        cycle_counter <= cycle_counter + 1;
        if (cycle_counter == 0) begin
          cycle_counter <= 0;  // stays at 0 for single-word-per-cycle mode
        end
        address_temp_2        <= address_temp_2 + 1;
        first_spad_data_delay <= first_spad_data_delay + 1'd1;

        // Update overhead accumulator and detect filter-boundary wrap.
        overhead_reg          <= overhead_reg + 2 + overhead_w;
        overhead_new_calc_reg <= overhead_reg;  // snapshot for delayed-write path
        if (overhead_reg + overhead_next_word >= filters_w) begin
          // Filter boundary: wrap overhead_reg by subtracting filters_w.
          overhead_reg <= 2 + overhead_reg + overhead_w - filters_w;
          if (overhead_reg == filters_w + 1) begin
            // Overshoot by exactly 1: apply extra correction.
            overhead_reg <= 2 + overhead_reg + overhead_w - filters_w - 1;
            over_ending  <= 1;
          end
          if (overhead_reg < filters_w) begin
            // Look-ahead: update new_calc for the delayed path as well.
            overhead_new_calc_reg <= overhead_reg + 2 + overhead_next_word;
          end
        end

        temp_acc_overhead     <= temp_acc_overhead + $bits(temp_acc_overhead)'(overhead_w);
        overhead_pos          <= overhead_pos + 2;

        // Wrap address_temp_2 and cycle_counter at SECOND_SPAD depth.
        if (address_temp_2 == SECOND_SPAD_ADDR - 1) begin
          address_temp_2 <= 0;
          cycle_counter  <= 0;
        end

        // Filter-boundary detection via next_channel_counter.
        if (next_channel_counter >= 0) begin
          data_storage_1    <= first_spad_data_delay;  // save reference level
          overhead_pos      <= 0;
          if (data_storage_1 >= 0) begin
            // Reset per-filter accumulator and advance the address shadow.
            temp_acc_overhead     <= 0;
            first_spad_addr_delay <= first_spad_addr_delay + 1;
          end
        end
      end else begin
        // Idle: seed data_storage_1 to -filters_w so the next data burst
        // sees the correct threshold from the very first word.
        data_storage_1     <= -filters_w;
        overhead_pos       <= $bits(overhead_pos)'(filters_w);
      end

      // -----------------------------------------------------------------------
      // compute_i: synchronous per-computation clear.
      // Resets SPAD-facing signals and internal accumulators.
      // Sets compute_sent so second_spad_words_o is cleared on the next
      // enable_i cycle (avoids a race with the write path).
      // Does NOT clear first_spad_addr_o or data_storage registers here;
      // those are cleared one cycle later via compute_delay.
      // -----------------------------------------------------------------------
      compute_delay <= compute_i;
      if (compute_i) begin
        second_spad_en_o      <= 0;
        first_spad_data_delay <= 0;
        address_temp_2        <= 0;
        cycle_counter         <= 0;
        second_spad_data_o    <= 0;
        overhead_reg          <= 0;
        overhead_new_calc_reg <= 0;
        overhead_delay_reg    <= 0;
        compute_sent          <= 1;  // flag: clear second_spad_words_o next enable cycle
        overhead_pos          <= 0;
      end

      // -----------------------------------------------------------------------
      // compute_delay: one-cycle-after-compute_i pipeline flush.
      // Clears the pipeline registers that could not be cleared in the
      // compute_i cycle itself (data_storage_1/2, address registers).
      // -----------------------------------------------------------------------
      if (compute_delay) begin
        data_storage_1        <= 0;
        data_storage_2        <= 0;
        first_spad_en_o       <= 0;
        first_spad_addr_delay <= 0;
        first_spad_addr_o     <= 0;
      end
    end
  end

endmodule
