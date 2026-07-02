// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// =============================================================================
// Module: data_pipeline_iact
// =============================================================================
// Purpose:
//   Implements the Input Activation (iact) data pipeline within a single PE
//   of the OpenEye sparse systolic array.  It receives a wide raw iact word
//   from the global iact bus, decomposes it into individual activation values,
//   detects non-zero values (sparse encoding), and distributes them across two
//   scratchpad (SPAD) memories with distinct organisations:
//
//     First SPAD  (iact address SPAD):
//       Stores the run-length / position encoding of each non-zero activation.
//       Each entry is FIRST_SPAD_DATA bits wide and represents a cumulative
//       count of non-zero values seen so far (the "overhead" counter), which
//       lets the MAC engine compute relative offsets without scanning zeros.
//
//     Second SPAD (iact data SPAD):
//       Stores the actual (non-zero) payload bytes alongside a 4-bit
//       transmission_counter tag that encodes the sub-word position of each
//       value inside the original DATA_WIDTH input word.
//       Word format: { transmission_counter_delay[3:0], payload[SECOND_PAYLOAD_WIDTH-1:0] }
//
// Sparsity encoding (SPARSITY_EN = 1, default):
//   Only non-zero DATA_WIDTH/SECOND_SPAD_DATA-wide sub-words are written to
//   the second SPAD.  The first SPAD records overhead = (number of non-zero
//   entries written so far) at each position, which the MAC unit uses to
//   decode relative positions without storing zeros.
//
// Dense mode (SPARSITY_EN = 0):
//   All sub-words are written regardless of zero/non-zero status.
//   (Currently the SPARSITY_EN parameter is present but the always block
//   does not gate the writes on it — future extension.)
//
// Uneven-ending handling:
//   When the input feature-map width is not a multiple of the word packing
//   factor (i.e. first_spad_max_i < 4), the last input word of each line
//   contains fewer valid activation bytes than a full SECOND_SPAD_DATA-wide
//   slot.  The module tracks this via uneven_ending / uneven_counter /
//   uneven_ending_storage and pre-loads cycle_counter to the correct offset
//   so the very next enable_i cycle resumes at the right sub-word position.
//
// Clock / Reset:
//   Single clock domain (clk_i).  Asynchronous active-low reset (rst_ni).
//   compute_i is a synchronous per-PE clear that resets all address and
//   data state but preserves and advances the uneven-ending bookkeeping.
//
// =============================================================================

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------
//
//  DATA_WIDTH                [default 24]
//    Width of the input data bus (data_i).  Must be a multiple of both
//    FIRST_SPAD_DATA and SECOND_SPAD_DATA.
//
//  FIRST_SPAD_ADDR           [default 16]
//    Depth (number of entries) of the first SPAD (address / overhead SPAD).
//
//  FIRST_SPAD_DATA           [default 4]
//    Width in bits of each entry in the first SPAD.
//    Also the granularity of the "overhead" counter (non-zero count per slot).
//
//  SECOND_SPAD_ADDR          [default 16]
//    Depth (number of entries) of the second SPAD (payload SPAD).
//
//  SECOND_SPAD_DATA          [default 12]
//    Full width in bits of each second SPAD entry:
//      = SECOND_OVERHEAD_WIDTH (4-bit tag) + SECOND_PAYLOAD_WIDTH (8-bit data).
//
//  SECOND_PAYLOAD_WIDTH      [default 8]
//    Width of the actual activation byte stored in the second SPAD.
//    Typically == DATA_IACT_BITWIDTH (8 for INT8).
//
//  SPARSITY_EN               [default 1]
//    When 1: sparse mode — only non-zero sub-words written to second SPAD.
//    When 0: dense mode — all sub-words written (future use).
//
//  FIRST_SPAD_ADDR_BITWIDTH  [= $clog2(FIRST_SPAD_ADDR)]
//    Address width for the first SPAD output port.
//
//  SECOND_SPAD_ADDR_BITWIDTH [= $clog2(SECOND_SPAD_ADDR)]
//    Address width for the second SPAD output port.
//
//  FIRST_SPAD_DATA_CYCLE     [= DATA_WIDTH / FIRST_SPAD_DATA]
//    Number of first-SPAD-sized sub-words packed into one DATA_WIDTH input.
//    e.g. 24 / 4 = 6 cycles to consume one 24-bit input word.
//
//  SECOND_SPAD_DATA_CYCLE    [= DATA_WIDTH / SECOND_SPAD_DATA]
//    Number of second-SPAD-sized sub-words packed into one DATA_WIDTH input.
//    e.g. 24 / 12 = 2 sub-words per input word.
//    Drives the cycle_counter wrap value.
//
//  SECOND_OVERHEAD_WIDTH     [= SECOND_SPAD_DATA - SECOND_PAYLOAD_WIDTH]
//    Width of the position tag field packed into each second SPAD entry.
//    Typically 4 bits (= transmission_counter_delay width).

// -----------------------------------------------------------------------------
// Port Descriptions
// -----------------------------------------------------------------------------
//
//  clk_i              - System clock.
//  rst_ni             - Asynchronous active-low reset.  Clears all state.
//  compute_i          - Synchronous per-computation clear pulse.  Resets SPAD
//                       addresses, data buffers, counters.  Also advances the
//                       uneven-line bookkeeping (uneven_counter).
//
//  data_i[DATA_WIDTH-1:0]
//                     - Wide raw iact input word.  On cycle_counter == 0 the
//                       full word is captured; on subsequent cycles the pipeline
//                       reads from the shifted data_storage_2 register.
//
//  enable_i           - When 1: module processes one sub-word per clock.
//                       When 0: module is idle; on the cycle immediately after
//                       enable_i falls (enable_delay_reg == 1) the final
//                       first_spad_words_o value is latched.
//
//  iact_x_line_repetitions_i[3:0]
//                     - Number of times each iact x-line is reused across
//                       cluster columns.  Used to determine when uneven_counter
//                       should wrap and toggle uneven_ending_storage.
//
//  first_spad_words_o[$clog2(FIRST_SPAD_ADDR+1)-1:0]
//                     - Number of valid entries written to the first SPAD.
//                       Updated on the falling edge of enable_i (one cycle
//                       after the last write) to first_spad_addr_o + 1.
//
//  first_spad_max_i[$clog2(FIRST_SPAD_ADDR)-1:0]
//                     - Maximum number of non-zero values per first-SPAD entry.
//                       Controls when first_spad_addr_o advances:
//                       address advances when transmission_counter >=
//                       data_storage_1 + first_spad_max_i.
//
//  second_spad_words_o[$clog2(SECOND_SPAD_ADDR+1)-1:0]
//                     - Count of valid (non-zero) entries written to the second
//                       SPAD.  Cleared on compute_i (via compute_sent flag).
//
//  first_spad_addr_o[FIRST_SPAD_ADDR_BITWIDTH-1:0]
//                     - Write address into the first SPAD.  Increments when
//                       the transmission_counter crosses the next threshold.
//
//  first_spad_data_o[FIRST_SPAD_DATA-1:0]
//                     - Data written to the first SPAD on each second-SPAD
//                       write: the current overhead_reg + 1 (running count
//                       of non-zero values stored so far in this slot).
//
//  first_spad_en_o    - Write-enable for the first SPAD.  Asserted every cycle
//                       while enable_i is high; also asserted for one cycle
//                       after enable_i falls to commit the final word count.
//
//  second_spad_addr_o[SECOND_SPAD_ADDR_BITWIDTH-1:0]
//                     - Write address into the second SPAD.  Driven from
//                       address_temp_2 (registered one cycle earlier).
//
//  second_spad_data_o[SECOND_SPAD_DATA-1:0]
//                     - Data written to the second SPAD.
//                       Combinational: { transmission_counter_delay[3:0],
//                                        payload_reg[SECOND_PAYLOAD_WIDTH-1:0] }
//                       The tag field encodes the sub-word position within the
//                       original DATA_WIDTH word.
//
//  second_spad_en_o   - Write-enable for the second SPAD.  Asserted only when
//                       a non-zero sub-word is detected (sparse gating).

module data_pipeline_iact #(
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
    parameter SECOND_SPAD_DATA_CYCLE    = DATA_WIDTH / SECOND_SPAD_DATA,
    parameter SECOND_OVERHEAD_WIDTH     = SECOND_SPAD_DATA - SECOND_PAYLOAD_WIDTH
) (
    input                                         clk_i,
    input                                         rst_ni,
    input                                         compute_i,
    //input                                         data_mode, Insert later

    input      [                DATA_WIDTH-1 : 0] data_i,
    input                                         enable_i,
    input     [                            3 : 0] iact_x_line_repetitions_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [   $clog2(FIRST_SPAD_ADDR)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [  FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output reg [           FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                    first_spad_en_o,

    output reg [ SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output     [          SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                    second_spad_en_o
);

  // ---------------------------------------------------------------------------
  // Internal Registers and Wires
  // ---------------------------------------------------------------------------

  // data_storage_1: running threshold for first_spad_addr advancement.
  //   Accumulates first_spad_max_i each time the first SPAD address steps;
  //   used to compare against transmission_counter so the address only
  //   increments once per "slot" of first_spad_max_i non-zero values.
  reg  [            FIRST_SPAD_DATA-1 : 0] data_storage_1;

  // data_storage_2: shift register holding the current input word.
  //   On cycle_counter == 0: loaded directly from data_i.
  //   On subsequent cycles: shifted right by SECOND_SPAD_DATA bits each cycle
  //   via the current_data combinational wire, exposing the next sub-word.
  reg  [                 DATA_WIDTH-1 : 0] data_storage_2;

  // address_temp_2: second SPAD write-address accumulator (one cycle ahead).
  //   Incremented each time a non-zero sub-word is written to the second SPAD.
  //   Registered into second_spad_addr_o one cycle later (pipeline register).
  //   Wraps to 0 when it reaches SECOND_SPAD_ADDR - 1.
  reg  [     $clog2(SECOND_SPAD_ADDR) : 0] address_temp_2;

  // cycle_counter: sub-word position counter within one DATA_WIDTH input word.
  //   Counts 0 .. SECOND_SPAD_DATA_CYCLE-1, then wraps.
  //   cycle_counter == 0: first sub-word; data is taken directly from data_i.
  //   cycle_counter != 0: subsequent sub-words; data taken from current_data.
  //   Seeded to a non-zero value by the uneven-ending logic when an input line
  //   does not end on a full-word boundary.
  reg  [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;

  // transmission_counter: global sub-word position counter (does not wrap).
  //   Incremented every enable_i cycle.  Compared against
  //   (data_storage_1 + first_spad_max_i) to decide when first_spad_addr_o
  //   should advance to the next first-SPAD slot.
  reg  [                          4-1 : 0] transmission_counter;

  // transmission_counter_delay: one-cycle delayed copy of transmission_counter.
  //   Packed into second_spad_data_o[SECOND_SPAD_DATA-1:SECOND_PAYLOAD_WIDTH]
  //   as the position tag so the MAC engine knows which weight position each
  //   non-zero value corresponds to.
  reg  [                          4-1 : 0] transmission_counter_delay;

  // overhead_reg: running count of non-zero second-SPAD entries written so far.
  //   Written into first_spad_data_o (as overhead_reg + 1) each time a
  //   non-zero sub-word is detected; tracks how many activations the MAC must
  //   accumulate before moving to the next first-SPAD address.
  reg  [      SECOND_OVERHEAD_WIDTH-1 : 0] overhead_reg;

  // enable_delay_reg: one-cycle delayed copy of enable_i.
  //   Used to detect the falling edge of enable_i so first_spad_words_o can
  //   be committed on the cycle after the last data word.
  reg                                      enable_delay_reg;

  // payload_reg: the SECOND_PAYLOAD_WIDTH-bit activation byte being staged
  //   for the second SPAD.  Loaded from data_i (cycle 0) or current_data
  //   (cycle > 0) whenever a non-zero sub-word is detected; output one cycle
  //   later via second_spad_data_o.
  reg  [       SECOND_PAYLOAD_WIDTH-1 : 0] payload_reg;

  // current_data: combinational right-shift of data_storage_2 by SECOND_SPAD_DATA.
  //   Exposes the next SECOND_SPAD_DATA-bit sub-word for cycle_counter > 0.
  wire [                 DATA_WIDTH-1 : 0] current_data;

  // compute_sent: single-cycle flag set on compute_i.
  //   On the next enable_i cycle after compute_i, ensures second_spad_words_o
  //   is cleared (handles the case where enable_i and compute_i coincide).
  reg                                      compute_sent;

  // uneven_ending: flag indicating that the current line starts mid-word.
  //   When set at the start of an enable_i burst, the first input sub-word
  //   is processed as a "leftover" from the previous (partial) line and is
  //   written to the second SPAD using data_i bits above SECOND_SPAD_DATA
  //   rather than the normal low bits.
  reg                                      uneven_ending;

  // uneven_ending_storage: persistent state of uneven_ending across compute_i.
  //   Toggled each time uneven_counter wraps around iact_x_line_repetitions_i,
  //   so alternating lines correctly start at sub-word 0 or sub-word 1.
  reg                                      uneven_ending_storage;

  // uneven_counter: counts how many compute_i pulses (iact x-line passes) have
  //   occurred since the last uneven_ending_storage toggle.
  //   Wraps at iact_x_line_repetitions_i - 1.
  reg  [                            3 : 0] uneven_counter;

  // ---------------------------------------------------------------------------
  // Combinational Logic
  // ---------------------------------------------------------------------------

  // current_data: right-shift data_storage_2 to reveal the next sub-word.
  // On cycle_counter == 0 this is not used (data taken from data_i directly).
  // On cycle_counter >= 1 this gives sub-word [cycle_counter] of the word.
  assign current_data = data_storage_2 >> SECOND_SPAD_DATA;

  // second_spad_data_o: packed second SPAD write word.
  //   Upper SECOND_OVERHEAD_WIDTH bits: transmission_counter_delay
  //     (sub-word position tag, one cycle delayed for pipeline alignment).
  //   Lower SECOND_PAYLOAD_WIDTH bits: payload_reg
  //     (the non-zero activation byte captured this cycle).
  assign second_spad_data_o = {
    transmission_counter_delay, payload_reg[SECOND_PAYLOAD_WIDTH-1 : 0]
  };

  // ---------------------------------------------------------------------------
  // Main Sequential Process
  // ---------------------------------------------------------------------------
  // Single always block handling reset, compute clear, and normal operation.
  //
  // Priority (highest to lowest):
  //   1. rst_ni == 0  -> asynchronous full reset of all registers.
  //   2. compute_i    -> synchronous per-computation clear (runs in parallel
  //                      with the enable_i path; takes effect at end of block).
  //   3. enable_i     -> normal data processing (one sub-word per cycle).
  //   4. !enable_i    -> idle; falling-edge detection via enable_delay_reg.
  //
  // Note: first_spad_en_o and second_spad_en_o default to 0 at the top of the
  // else branch so they produce one-cycle pulses without explicit clearing.
  // ---------------------------------------------------------------------------
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      // -----------------------------------------------------------------------
      // Asynchronous Reset
      // Clears all outputs and internal state unconditionally.
      // -----------------------------------------------------------------------
      first_spad_words_o         <= 0;
      second_spad_words_o        <= 0;
      first_spad_data_o          <= 0;
      first_spad_addr_o          <= 0;
      first_spad_en_o            <= 0;
      second_spad_addr_o         <= 0;
      second_spad_en_o           <= 0;
      data_storage_1             <= 0;
      data_storage_2             <= 0;
      address_temp_2             <= 0;
      cycle_counter              <= 0;
      payload_reg                <= 0;
      overhead_reg               <= 0;
      enable_delay_reg           <= 0;
      transmission_counter       <= 0;
      transmission_counter_delay <= 0;
      compute_sent               <= 0;
      uneven_ending              <= 0;
      uneven_ending_storage      <= 0;
      uneven_counter             <= 0;
    end else begin
      // -----------------------------------------------------------------------
      // Default: de-assert one-cycle pulse outputs.
      // Both SPAD enables default to 0; set to 1 below only when a write occurs.
      // -----------------------------------------------------------------------
      first_spad_en_o   <= 0;
      second_spad_en_o  <= 0;

      // Register enable_i for falling-edge detection.
      enable_delay_reg <= enable_i;

      if (enable_i == 1) begin
        // ---------------------------------------------------------------------
        // Active data-processing cycle.
        // Executed once per clock while enable_i is high.
        // One SECOND_SPAD_DATA-wide sub-word is processed per cycle.
        // ---------------------------------------------------------------------

        // Assert first SPAD write enable unconditionally while data is arriving.
        // The address / data driven to the first SPAD are updated below.
        first_spad_en_o <= 1;

        // Clear the compute_sent flag and uneven_ending so they do not persist
        // into the active processing burst.
        compute_sent    <= 0;
        uneven_ending   <= 0;

        // On the first enable_i cycle after compute_i, clear second_spad_words_o.
        // This is deferred by one cycle via compute_sent to avoid a race between
        // the compute_i reset path and the enable_i increment path.
        if (compute_sent) begin
          second_spad_words_o <= 0;
        end

        // Advance the sub-word position counters.
        cycle_counter              <= cycle_counter + 1;
        transmission_counter       <= transmission_counter + 1;
        transmission_counter_delay <= transmission_counter; // 1-cycle pipeline delay

        // First SPAD address advancement:
        // When the total number of processed sub-words (transmission_counter)
        // reaches the next threshold (data_storage_1 + first_spad_max_i),
        // step to the next first-SPAD address slot and update the threshold.
        if (transmission_counter >= (data_storage_1 + first_spad_max_i)) begin
          data_storage_1    <= data_storage_1 + first_spad_max_i;
          first_spad_addr_o <= first_spad_addr_o + 1;
        end

        // Wrap cycle_counter at SECOND_SPAD_DATA_CYCLE so it stays in range.
        if (cycle_counter == SECOND_SPAD_DATA_CYCLE - 1) begin
          cycle_counter <= 0;
        end

        // Pipeline: latch address_temp_2 (computed last cycle) into the output
        // port; shift the data register to prepare the next sub-word.
        second_spad_addr_o  <= address_temp_2[SECOND_SPAD_ADDR_BITWIDTH-1:0];
        data_storage_2      <= current_data;  // shift right by SECOND_SPAD_DATA

        // -----------------------------------------------------------------
        // Sparse write to second SPAD:
        // Write only when the current sub-word is non-zero.
        //
        // Two cases:
        //   cycle_counter == 0: first sub-word of a new DATA_WIDTH word;
        //     check data_i[SECOND_PAYLOAD_WIDTH-1:0] (low byte of raw input).
        //   cycle_counter != 0: subsequent sub-words;
        //     check current_data (shifted window into data_storage_2).
        //
        // On a non-zero detect:
        //   - Increment second_spad_words_o (count of stored non-zero values).
        //   - Write overhead_reg + 1 to first_spad_data_o (cumulative count
        //     used by the MAC to decode relative positions).
        //   - Increment overhead_reg for the next write.
        //   - Assert second_spad_en_o (one-cycle pulse).
        //   - Advance address_temp_2 (takes effect at second_spad_addr_o next cycle).
        //   - Wrap address_temp_2 to 0 at SECOND_SPAD_ADDR boundary; also
        //     reset cycle_counter to restart sub-word parsing.
        // -----------------------------------------------------------------
        if (((cycle_counter == 0) & (data_i[SECOND_PAYLOAD_WIDTH-1:0] != 0)) |
            ((cycle_counter != 0) & (current_data != 0))) begin
          second_spad_words_o <= second_spad_words_o + 1;
          first_spad_data_o   <= overhead_reg + 1'd1;
          overhead_reg        <= overhead_reg + 1;
          second_spad_en_o    <= 1;
          address_temp_2      <= address_temp_2 + 1;
          if (address_temp_2 == SECOND_SPAD_ADDR - 1) begin
            address_temp_2 <= 0;
            cycle_counter  <= 0;
          end
        end

        // -----------------------------------------------------------------
        // Payload staging: capture the payload byte for the current sub-word.
        //
        // cycle_counter == 0 (first sub-word):
        //   Load payload_reg from data_i[SECOND_PAYLOAD_WIDTH-1:0] (low byte).
        //   Also reload data_storage_2 from data_i so subsequent cycles can
        //   shift through it.  Guard: only when data_i != 0 (any byte non-zero).
        //
        // cycle_counter != 0 (subsequent sub-words):
        //   Load payload_reg from current_data[SECOND_PAYLOAD_WIDTH-1:0].
        //   Guard: only when current_data != 0.
        //
        // payload_reg is output via second_spad_data_o combinationally and
        // will be captured by the SPAD on the cycle second_spad_en_o is high.
        // -----------------------------------------------------------------
        if ((cycle_counter == 0) & (data_i != 0)) begin
          payload_reg       <= data_i[SECOND_PAYLOAD_WIDTH-1 : 0];
          data_storage_2    <= data_i;
        end
        if ((cycle_counter != 0) & (current_data != 0)) begin
          payload_reg         <= current_data[SECOND_PAYLOAD_WIDTH-1 : 0];
        end

        // -----------------------------------------------------------------
        // Uneven-ending sub-word handling:
        // When uneven_ending is set (line started mid-word), the first
        // enable_i cycle carries a "leftover" byte from the previous line.
        // It is extracted from data_i bits above SECOND_SPAD_DATA (i.e.
        // the high byte of the first input word when only one byte remains
        // from the previous line).
        //
        // This leftover byte is always treated as non-zero for second-SPAD
        // purposes (uneven_ending is only set when such a byte exists).
        // -----------------------------------------------------------------
        if (uneven_ending) begin
          second_spad_words_o <= second_spad_words_o + 1;
          payload_reg         <= data_i[SECOND_SPAD_DATA + SECOND_PAYLOAD_WIDTH-1 : SECOND_SPAD_DATA];
          second_spad_en_o    <= 1;
          address_temp_2      <= address_temp_2 + 1;
          overhead_reg        <= overhead_reg + 1;
          first_spad_data_o   <= overhead_reg + 1'd1;
        end

      end else begin
        // ---------------------------------------------------------------------
        // Idle: enable_i == 0.
        // On the first idle cycle after an active burst (enable_delay_reg == 1),
        // commit the final first_spad_words_o value and assert first_spad_en_o
        // for one cycle so the SPAD latches the last address count.
        //
        // first_spad_words_o = first_spad_addr_o + 1
        //   (total number of first-SPAD entries written in this burst).
        // ---------------------------------------------------------------------
        if (enable_delay_reg == 1) begin
          first_spad_en_o    <= 1;
          first_spad_words_o <= first_spad_addr_o + 1;
        end
      end

      if (compute_i) begin
        // ---------------------------------------------------------------------
        // Synchronous per-computation clear (compute_i pulse).
        // Resets all SPAD-facing signals and internal data/address state
        // in preparation for the next iact delivery batch.
        //
        // Does NOT clear first_spad_words_o or second_spad_words_o here;
        // second_spad_words_o is cleared on the next enable_i cycle via
        // compute_sent; first_spad_words_o is updated at enable_i falling edge.
        //
        // Uneven-ending bookkeeping:
        //   When first_spad_max_i < 4 (line width is not a multiple of the
        //   packing factor), the next line may need to start at sub-word 1
        //   instead of 0.  The logic here:
        //   - Increments uneven_counter; wraps at iact_x_line_repetitions_i - 1.
        //   - When counter wraps: toggles uneven_ending_storage and seeds
        //     cycle_counter and uneven_ending to 1 - uneven_ending_storage
        //     (next line starts mid-word).
        //   - Otherwise: seeds them from uneven_ending_storage (continue
        //     alternating pattern).
        //   When compute_sent is already set (two consecutive compute_i pulses):
        //     fully resets cycle_counter, uneven_ending, uneven_counter to 0.
        // ---------------------------------------------------------------------
        first_spad_en_o            <= 0;
        second_spad_en_o           <= 0;
        data_storage_1             <= 0;
        data_storage_2             <= 0;
        first_spad_data_o          <= 0;
        address_temp_2             <= 0;
        first_spad_addr_o          <= 0;
        payload_reg                <= 0;
        overhead_reg               <= 0;
        transmission_counter       <= 0;
        transmission_counter_delay <= 0;
        second_spad_addr_o         <= 0;
        second_spad_words_o        <= 0;
        compute_sent               <= 1;  // flag: clear second_spad_words_o next enable cycle
        cycle_counter              <= 0;

        // Uneven-line handling (only for non-FC layers where lines don't fill full words)
        if (first_spad_max_i < 2) begin
          uneven_counter <= uneven_counter + 1;
          if (uneven_counter == iact_x_line_repetitions_i - 1) begin
            uneven_counter <= 0;
          end
          if (uneven_counter == iact_x_line_repetitions_i - 1) begin
            // Wrap: toggle the storage flag; next line starts at the opposite sub-word.
            uneven_ending_storage <= 1 - uneven_ending_storage;
            cycle_counter         <= 1 - uneven_ending_storage;
            uneven_ending         <= 1 - uneven_ending_storage;
          end else begin
            // No wrap: continue with the current storage value.
            cycle_counter <= uneven_ending_storage;
            uneven_ending <= uneven_ending_storage;
          end
        end

        // Fully reset uneven state on a second consecutive compute_i pulse.
        if (compute_sent) begin
          cycle_counter  <= 0;
          uneven_ending  <= 0;
          uneven_counter <= 0;
        end
      end
    end
  end

endmodule
