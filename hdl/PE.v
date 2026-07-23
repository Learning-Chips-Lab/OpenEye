// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE
///
/// The PE (Processing Element) is a fundamental computational unit in the OpenEye neural network
/// accelerator. It is designed to efficiently perform multiply-accumulate (MAC) operations for
/// neural network inference, with specific optimizations for handling sparse data patterns and
/// configurable fixed-point arithmetic.
///
/// Systolic Array Context:
/// The PE is designed to be instantiated in a 2D systolic array architecture where multiple PEs
/// work cooperatively to accelerate neural network inference. Key aspects of PE interconnection:
///
/// - Array Topology: PEs are organized in an X-Y grid (coordinates specified by PE_X, PE_Y)
/// - Data Flow Patterns:
///   * Input Activations: Can be sourced from multiple global buffers (NUM_GLB_IACT) or from
///     neighboring PEs. The iact_select_i signal determines which source is active.
///   * Weights: Flow through the array in a stationary or streaming pattern depending on the
///     dataflow configuration (weight-stationary, output-stationary, or row-stationary).
///   * Partial Sums: Flow between PEs in a systolic manner. Each PE receives partial sums from
///     its predecessor (psum_data_i), accumulates them with local MAC results, and forwards
///     them to the next PE (psum_data_o).
///
/// - Inter-PE Communication:
///   * Handshake Protocol: Uses valid-ready signaling for flow control (enable/ready pairs)
///   * Backpressure: PEs can stall the pipeline when downstream is not ready
///   * Broadcast: Input activations can be broadcast across multiple PEs simultaneously
///   * Multicast: Weights can be shared across PEs in the same column/row
///
/// Architecture:
/// - Memory Hierarchy: Uses a combination of scratch pads (SPads) for input activations (Iact),
///   weights (Wght), and partial sums (Psum) to maximize data reuse and minimize memory access.
///   * Iact SPad: Two-level structure with address SPad (indirect addressing) and data SPad
///   * Wght SPad: Two-level structure with address SPad and data SPad for sparse storage
///   * Psum SPad: Dual-ported memory allowing simultaneous read-accumulate-write operations
///
/// - Sparsity Exploitation: Implements zero-skipping logic for both input activations and weights
///   to avoid unnecessary computations on zero values.
///   * Compressed Sparse Format: Data is stored with overhead bits indicating zero positions
///   * Dynamic Indexing: Uses sparsity metadata to skip over zeros during MAC operations
///   * Zero Detection: Hardware detects and skips multiplications involving zero operands
///
/// - Parallel Processing: Supports parallel MAC operations through dual multipliers and adders.
///   * Configurable Parallelism: PARALLEL_MACS parameter controls number of simultaneous MACs
///   * Dual Datapath: Two independent multiply-accumulate units operating concurrently
///   * Serial Fallback: Can operate in serial mode (SERIAL parameter) for area optimization
///
/// - Flexible Precision: Configurable fixed-point arithmetic to balance accuracy and efficiency.
///   * Separate Bitwidths: Independent control of iact, weight, and psum precision
///   * Fractional Bits: Configurable fraction_bit_reg for fixed-point representation
///   * Accumulator Width: Wider psum bitwidth (default 20-bit) prevents overflow
///
/// - Data Flow Control: Uses a sophisticated FSM to coordinate data movement and computation.
///
/// Operational Flow:
/// 1. Configuration Phase:
///    - Parameter streaming: Receives stride, filter count, channel count via data_stream_i
///    - FSM states: FIRST_PARAMS -> SECOND_PARAMS -> THIRD_PARAMS -> FOURTH_PARAMS
///    - Configures operational parameters: filters_reg_M0, channel_reg_C0, iact_addr_max_reg
///
/// 2. Memory Loading Phase:
///    - Input activations and weights are loaded into respective SPad memories via data pipelines
///    - Memory addressing structures are initialized for sparse data processing
///    - Handshaking: Uses enable/ready signals to coordinate with external memory controllers
///    - Dual-buffer Loading: Separate pipelines for address and data SPads allow parallel loading
///
/// 3. Computation Phase (Detailed State Machine):
///    - IDLE: Waiting for compute_i trigger signal, all memories loaded and ready
///    - LOADING_1-5: Five-stage pipeline fill sequence
///      * LOADING_1: Initialize SPad addresses, assert read enables
///      * LOADING_2: First data becomes available from SPads
///      * LOADING_3: Setup multiplier inputs, prepare adder pipeline
///      * LOADING_4: Multiplier results available, read existing partial sums
///      * LOADING_5: Adders ready, prepare write-back to psum SPad
///    - CALCULATING: Main computation loop
///      * Reads next iact from Iact SPad (uses iact_addr_SPad_addr as pointer)
///      * Uses iact overhead bits to determine weight address range
///      * Fetches corresponding weights from Wght SPad (indexed by sparsity metadata)
///      * Reads existing partial sums from Psum SPad (dual-port simultaneous access)
///      * Multiplies: iact × weight in parallel multipliers
///      * Accumulates: MAC result + existing psum in parallel adders
///      * Writes back: Updated psums to Psum SPad (with hazard detection)
///      * Increments: iact_addr_current pointer, repeats until iact_addr_max_reg reached
///    - WAIT_TO_SEND_PSUM: Computation complete, waiting for psum_enable_i to stream results
///    - SEND_PSUM: Streaming accumulated partial sums to next PE or output
///      * Sequential readout of Psum SPad contents
///      * Valid-ready handshaking with downstream PE
///      * Returns to IDLE when psum_enable_i deasserts
///
/// 4. Output Phase:
///    - Accumulates results across multiple operations
///    - Manages partial sum routing and accumulation
///    - Coordinates output streaming of completed results
///    - Psum forwarding: Results flow to next PE in systolic chain
///
/// Sparse Data Handling (Detailed Example):
/// The PE uses a compressed sparse format to skip zero values efficiently:
///
/// Example: Computing Y = A × W where A = [0, 3, 0, 0, 5, 0, 2] and W is a sparse weight matrix
///
/// 1. Sparse Encoding:
///    - Iact Data SPad stores: [3, 5, 2] (non-zero values only)
///    - Iact Addr SPad stores: [1, 4, 6] (positions of non-zeros)
///    - Iact Overhead: [2, 3, 1] (gaps between non-zeros: 1→4 gap=3, 4→6 gap=2, start gap=2)
///
/// 2. Weight Indexing:
///    - Wght Addr SPad: Contains base addresses for each iact index
///    - Wght Data SPad: Contains packed weights with ignore_zeros metadata
///    - For iact[1]=3: reads weights from W[1,:] using base_addr + sparsity offset
///    - For iact[4]=5: reads weights from W[4,:] using base_addr + sparsity offset
///
/// 3. Zero-Skipping Execution:
///    Cycle 1: Read iact_addr=1, fetch iact_data=3
///            Use overhead=2 to skip first 2 weight positions
///            MAC: 3 × W[1,2], 3 × W[1,3] (parallel MACs)
///    Cycle 2: Read iact_addr=4, fetch iact_data=5
///            Use overhead=3 to advance weight pointer
///            MAC: 5 × W[4,5], 5 × W[4,6] (parallel MACs)
///    Cycle 3: Read iact_addr=6, fetch iact_data=2
///            Use overhead=1 to advance weight pointer
///            MAC: 2 × W[6,7], 2 × W[6,8] (parallel MACs)
///
/// 4. Partial Sum Accumulation:
///    - First cycle: psum_spad reads return 0 (first use), MAC results written to psum[0:1]
///    - Later cycles: psum_spad reads return previous accumulations, add to new MACs
///    - Hazard detection: reuse_psum_spad_a/b flags handle read-after-write on same address
///
/// Interface Timing Diagrams and Protocols:
///
/// Input Activation Interface (AXI-Stream-like):
///    clk     : __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__
///    iact_data_i  : ====< D0 >===< D1 >===< D2 >===
///    iact_enable_i: ________|‾‾‾‾‾‾‾‾|___|‾‾‾‾‾‾‾‾|___
///    iact_ready_o : ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|_________|‾‾
///    - Valid-ready handshake: Transfer occurs when both enable and ready are high
///    - Backpressure: ready_o deasserts when SPad is full
///    - Multi-source: iact_select_i chooses from NUM_GLB_IACT input sources
///
/// Weight Interface:
///    clk     : __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__
///    wght_data_i  : ====< W0 >===< W1 >===< W2 >===
///    wght_enable_i: ________|‾‾‾‾‾‾‾‾|___|‾‾‾‾‾‾‾‾|___
///    wght_ready_o : ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|_________|‾‾
///    - Similar handshaking to iact interface
///    - Weights loaded during idle/configuration phase
///    - Can be broadcast to multiple PEs in weight-stationary dataflow
///
/// Partial Sum Interface (Systolic Data Flow):
///    clk     : __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__
///    psum_data_i  : ====< P0 >===< P1 >===< P2 >===  (from previous PE)
///    psum_enable_i: ________|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|___
///    psum_ready_o : ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
///    psum_data_o  : =========< Q0 >===< Q1 >===  (to next PE)
///    psum_enable_o: ______________|‾‾‾‾‾‾‾‾‾‾‾‾|___
///    psum_ready_i : ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
///    - Bidirectional: Receives psums from upstream, sends to downstream
///    - Systolic flow: Data ripples through PE array in pipelined fashion
///    - Internal accumulation: Incoming psums added to local MAC results
///
/// Control Interface:
///    clk     : __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__
///    compute_i    : ____________|‾‾‾|_________________________
///    (PE transitions from IDLE → LOADING_1 → ... → CALCULATING)
///    - Single-cycle pulse triggers computation start
///    - Must assert only when iact_set and wght_set flags are both high
///
/// Configuration Streaming:
///    clk     : __|‾‾|__|‾‾|__|‾‾|__|‾‾|__
///    enable_stream_i: __|‾‾‾|___|‾‾‾|___|‾‾‾|___
///    data_stream_i: ==< PARAM1 >< PARAM2 >< PARAM3 >=
///    (FSM: FIRST_PARAMS → SECOND_PARAMS → THIRD_PARAMS)
///    - Sequential parameter streaming
///    - Each enable pulse advances FSM and latches 12-bit parameter
///
/// Key Features:
/// - Zero-skipping optimization for sparse data (up to 10× speedup on 90% sparse data)
/// - Parallel MAC operations for improved throughput (2× with PARALLEL_MACS=2)
/// - Configurable fixed-point arithmetic (8/16-bit activations, 8/16-bit weights, 20-bit psums)
/// - Dual-ported memory architecture for efficient data access (simultaneous read/write)
/// - Flexible routing for input activations and partial sums (systolic and broadcast modes)
/// - State machine controlled operation for precise timing
/// - Read-after-write hazard detection with data forwarding (reuse_psum_spad_a/b logic)
/// - Pipeline depth: 5 cycles from data load to MAC result writeback
///
/// Performance Optimizations:
/// - Efficient memory hierarchy to minimize data movement (on-chip SPads vs. DRAM access)
/// - Parallel processing units for increased throughput (dual MACs achieve 2 ops/cycle)
/// - Sparsity exploitation to skip unnecessary computations (zero detection in hardware)
/// - Pipelined operation for sustained performance (overlapped SPad access and computation)
/// - Data reuse: Weights and activations stay in local SPads across multiple MAC operations
/// - Bypass network: Forwards freshly computed psums to avoid SPad read latency
///
/// Parameters:
/// Configuration Parameters:
///    IS_TOPLEVEL             - Boolean flag to indicate if this module is the top level
///                              0: PE is instantiated within a larger array (default behavior)
///                              1: PE is top-level module (enables additional debug features)
///    SERIAL                  - Boolean flag to enable serial processing mode
///                              0: Parallel mode - dual MACs operate simultaneously (default)
///                              1: Serial mode - single MAC, time-multiplexed operation
///                              Serial mode reduces area at the cost of 2× latency
///    PARALLEL_MACS           - Number of multiply-accumulate operations executed in parallel
///                              Typical values: 1 (serial), 2 (default dual-MAC), 4 (quad-MAC)
///                              Affects throughput (ops/cycle) and area (multiplier count)
///                              Must match WGHT_DATA_DATA packing format
///    SPARSITY_EN             - Sparsity optimization enable flag
///                              1: Sparse mode (default) - enables zero-skipping via overhead bits
///                              0: Dense mode - sequential addressing, no overhead processing
///                              Affects FSM selection, SPad instantiation, and data unpacking logic
///    USE_DSP                 - MAC implementation selection
///                              0: Standard multiplier + adder pipeline (fabric-based, default)
///                              1: DSP48 slice optimization (integrated MAC, FPGA-optimized)
///                              Controls which compute unit generate block is instantiated
///    DATA_IACT_OVERHEAD      - Bits reserved for zero-skipping metadata in input activations
///                              Default: 4 bits (can encode gaps of 0-15 zeros)
///                              Total iact word width = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD
///                              Larger overhead supports sparser activations but increases memory
///    DATA_WGHT_IGNORE_ZEROS  - Bits reserved for zero-skipping metadata in weights
///                              Default: 4 bits per weight (encodes position within sparse structure)
///                              Used as offset into psum SPad: psum_addr = base + ignore_zeros
///                              Allows efficient indexing of non-zero weight positions
///
/// Data Width Parameters:
///    DATA_IACT_BITWIDTH      - Bit width of input activation values (payload only)
///                              Typical: 8 (INT8), 16 (FP16/INT16), 4 (INT4 quantization)
///                              Does NOT include overhead bits for sparsity metadata
///    DATA_WGHT_BITWIDTH      - Bit width of weight values (payload only)
///                              Typical: 8 (INT8 quantized), 16 (higher precision)
///                              Should match iact bitwidth for balanced precision
///    DATA_PSUM_BITWIDTH      - Bit width of partial sum accumulator
///                              Default: 20 bits to prevent overflow during accumulation
///                              Must be >= (IACT_BITWIDTH + WGHT_BITWIDTH + log2(max_accumulations))
///                              Wider psums trade area for numerical accuracy
///
/// Sparsity Parameters:
///    DATA_IACT_OVERHEAD      - Bits reserved for zero-skipping metadata in input activations
///                              Default: 4 bits (can encode gaps of 0-15 zeros)
///                              Total iact word width = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD
///                              Larger overhead supports sparser activations but increases memory
///    DATA_WGHT_IGNORE_ZEROS  - Bits reserved for zero-skipping metadata in weights
///                              Default: 4 bits per weight (encodes position within sparse structure)
///                              Used as offset into psum SPad: psum_addr = base + ignore_zeros
///                              Allows efficient indexing of non-zero weight positions
///
/// Memory Organization Parameters:
///    IACT_DATA_ADDR          - Depth of input activation data scratch pad memory
///                              Default: 16 entries (stores non-zero activation values)
///                              Size should match maximum activations per PE per layer
///                              Each entry: (DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD) bits wide
///    IACT_ADDR_ADDR          - Depth of input activation address scratch pad memory
///                              Default: 9 entries (stores indices of non-zero activations)
///                              Smaller than IACT_DATA_ADDR due to compression
///                              Each entry: $clog2(IACT_DATA_ADDR) bits wide
///    WGHT_DATA_ADDR          - Depth of weight data scratch pad memory
///                              Default: 96 entries (stores weight values for filter kernel)
///                              Must accommodate all weights for assigned filters
///                              Each entry: (DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS bits
///    WGHT_ADDR_ADDR          - Depth of weight address scratch pad memory
///                              Default: 16 entries (stores base addresses for weight lookup)
///                              Indirection layer for sparse weight access
///                              Each entry: $clog2(WGHT_DATA_ADDR) bits wide
///    PSUM_ADDR               - Depth of partial sum scratch pad memory
///                              Default: 32 entries (accumulates intermediate results)
///                              Dual-ported for simultaneous read-modify-write
///                              Each entry: DATA_PSUM_BITWIDTH bits wide
///                              Size determines maximum output feature map size per PE
///
/// Interface Parameters:
///    TRANS_BITWIDTH_IACT     - Bit width of input activation interface bus
///                              Default: 24 bits (flexible packing)
///                              Can carry: 3×8-bit data OR 2×12-bit data OR 6×4-bit addresses
///                              Wider bus amortizes transfer overhead
///    TRANS_BITWIDTH_WGHT     - Bit width of weight interface bus
///                              Default: 24 bits (flexible packing)
///                              Can carry: 3×8-bit weights OR 2×12-bit weights OR 3×8-bit addresses
///                              Should match GLB interface width for efficient streaming
///    NUM_GLB_IACT            - Number of global input activation buffer interfaces
///                              Default: 3 (multicast from 3 separate GLB banks)
///                              Allows PE to select from multiple activation sources
///                              iact_select_i chooses active source (0 to NUM_GLB_IACT-1)
///
/// Ports:
/// Clock and Reset:
///    clk_i                   - System clock input (positive edge triggered)
///                              All registers update on rising edge of clk_i
///                              Typical frequency: 100-500 MHz depending on target technology
///    rst_ni                  - Active-low asynchronous reset
///                              Assert low to reset all state machines and registers to initial state
///                              Clears all SPad contents, resets FSMs to IDLE
///
/// Input Activation Interface:
///    iact_select_i          - Input activation source selection control
///                              Width: $clog2(NUM_GLB_IACT+1) bits
///                              Range: 0 to NUM_GLB_IACT (0 disables, 1-NUM_GLB_IACT selects source)
///                              Determines which GLB iact interface is active
///                              Used for multicast routing in systolic array
///    iact_data_i            - Input activation data bus [includes value, sparsity bits, address]
///                              Width: TRANS_BITWIDTH_IACT * NUM_GLB_IACT bits (concatenated sources)
///                              Format: Packed data or addresses depending on load phase
///                              Selected by iact_select_i, routed through internal multiplexer
///    iact_enable_i          - Input activation data valid signal (per-source)
///                              Width: NUM_GLB_IACT bits (one per GLB source)
///                              High indicates valid data on corresponding iact_data_i slice
///                              Part of valid-ready handshake protocol
///    iact_ready_o           - Input activation interface ready signal (per-source)
///                              Width: NUM_GLB_IACT bits (one per GLB source)
///                              High indicates PE can accept new iact data
///                              Deasserts when iact SPads are full or during computation
///
/// Weight Interface:
///    wght_data_i            - Weight data bus [includes value, sparsity bits, address]
///                              Width: TRANS_BITWIDTH_WGHT bits
///                              Format: Packed weight data or addresses for SPad loading
///                              Feeds data_pipeline_wght module for unpacking and storage
///    wght_enable_i          - Weight data valid signal
///                              High indicates valid data present on wght_data_i
///                              Transfer occurs when both wght_enable_i and wght_ready_o are high
///    wght_ready_o           - Weight interface ready signal
///                              High indicates PE can accept new weight data
///                              Controlled by FSM: high during IDLE, low during computation
///
/// Partial Sum Interface:
///    psum_data_i            - Partial sum input data bus (from upstream PE)
///                              Width: TRANS_BITWIDTH_PSUM bits
///                              = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER
///                              Carries accumulated partial sums from previous PE in systolic chain
///                              Can be added to local MAC results during accumulation
///    psum_enable_i          - Partial sum input valid signal
///                              High indicates valid psum data from upstream PE
///                              Used during output phase to stream results out
///                              When high, PE reads from psum SPad and sends to output
///    psum_ready_o           - Partial sum input interface ready signal
///                              High indicates PE is ready to accept incoming partial sums
///                              Gated by internal psum_select signal
///                              Part of backpressure mechanism in systolic array
///    psum_data_o            - Partial sum output data bus (to downstream PE)
///                              Width: TRANS_BITWIDTH_PSUM bits
///                              Streams accumulated results to next PE or output buffer
///                              Data comes from psum SPad during SEND_PSUM state
///    psum_enable_o          - Partial sum output valid signal
///                              High indicates valid psum data being sent downstream
///                              Asserted during SEND_PSUM state
///                              Part of valid-ready handshake with next PE
///    psum_ready_i           - Partial sum output interface ready signal
///                              High indicates downstream PE/buffer can accept psum data
///                              Transfer occurs when both psum_enable_o and psum_ready_i are high
///                              Backpressure: PE stalls if downstream not ready
///
/// Control Interface:
///    compute_i              - Computation start trigger signal
///                              Single-cycle pulse to initiate MAC computation
///                              Transitions FSM from IDLE → LOADING_1
///                              Only valid when iact_set and wght_set flags are both high
///                              Typically asserted by global controller after data loading
///    enable_stream_i        - Parameter stream enable signal
///                              Each pulse advances configuration FSM and latches parameter
///                              Used during initialization to configure PE operation
///                              Sequence: FIRST_PARAMS → SECOND_PARAMS → THIRD_PARAMS
///    data_stream_i          - Configuration parameter data stream
///                              Width: 9 bits
///                              Carries runtime configuration: stride, filters, channels, etc.
///                              Bit assignments: [8:4]=filters, [7:4]=iact_x_line_repetitions,
///                              [3:0]=channel count/iact_addr_max
///                              Format varies by config FSM state (see configuration phase above)
///
/// FSM State Transitions and Descriptions:
///
/// Configuration Streaming FSM (current_state_stream):
///    State 0: FIRST_PARAMS  - Receives stride[3:1], wght_addr_max[7:4]
///                             Waits for enable_stream_i pulse
///                             Next: SECOND_PARAMS
///    State 1: SECOND_PARAMS - Receives filters[8:4], channels[3:0]
///                             Next: THIRD_PARAMS (if enable_stream_i)
///                             Timeout: FIRST_PARAMS (if not enabled within window)
///    State 2: THIRD_PARAMS  - Receives iact_addr_max configuration
///                             Next: FOURTH_PARAMS
///    State 3: FOURTH_PARAMS - Final configuration state (reserved for future expansion)
///                             Next: Returns to FIRST_PARAMS for reconfiguration
///
/// Main Computation FSM (current_state_computing):
///    State 0: IDLE          - Initial state after reset or completion
///                             Waiting for data loading to complete (iact_set && wght_set)
///                             Ready signals high (iact_ready_o, wght_ready_o = 1)
///                             All SPad addresses reset to 0
///                             Transitions: compute_i pulse → LOADING_1
///                                         psum_enable_i high → WAIT_TO_SEND_PSUM (bypass mode)
///
///    State 1: LOADING_1     - Pipeline fill cycle 1
///                             Initialize SPad read addresses (iact_addr, wght_addr = 0)
///                             Assert SPad read enables (iact_data_SPad_en_r = 1)
///                             Purpose: Start memory read operations
///                             Duration: 1 cycle
///                             Transitions: Unconditional → LOADING_2
///
///    State 2: LOADING_2     - Pipeline fill cycle 2
///                             SPad data appears on outputs (1 cycle read latency)
///                             Load first iact value into pipeline (iact_data_current_1)
///                             Begin weight address computation
///                             Duration: 1 cycle
///                             Transitions: Unconditional → LOADING_3
///
///    State 3: LOADING_3     - Pipeline fill cycle 3
///                             Iact propagates through pipeline (→ iact_data_current_2)
///                             Weight data ready from SPad
///                             Setup multiplier inputs (mult_1_fac_1, mult_1_fac_2)
///                             Duration: 1 cycle
///                             Transitions: Unconditional → LOADING_4
///
///    State 4: LOADING_4     - Pipeline fill cycle 4
///                             Iact reaches multipliers (→ iact_data_current_3)
///                             Multiplier results computed (combinational)
///                             Read existing partial sums from psum SPad
///                             Check used_psum_memory bitmap for initialization
///                             Duration: 1 cycle
///                             Transitions: Unconditional → LOADING_5
///
///    State 5: LOADING_5     - Pipeline fill cycle 5
///                             Multiplier outputs valid (mult_1_o_w, mult_2_o_w)
///                             Adder inputs setup: psum + MAC result
///                             Prepare writeback addresses (psum_spad_addr_a_w, _b_w)
///                             Hazard detection: Check for read-after-write conflicts
///                             Duration: 1 cycle
///                             Transitions: Unconditional → CALCULATING
///
///    State 6: CALCULATING   - Main computation loop (steady state)
///                             Concurrent operations each cycle:
///                               1. Write: Previous cycle's results → psum SPad
///                               2. Read: Next iact from iact_data_SPad
///                               3. Compute: Current iact × weights (dual MACs)
///                               4. Accumulate: MAC results + existing psums
///                               5. Increment: iact_addr_current pointer
///                             Pipeline operation: 5 operations in flight simultaneously
///                             Zero-skipping: Uses iact/wght overhead to skip zeros
///                             Duration: Variable (until all iacts processed)
///                             Exit condition: iact_addr_current == iact_addr_max_reg
///                             Transitions: All iacts done → WAIT_TO_SEND_PSUM
///
///    State 7: WAIT_TO_SEND_PSUM - Computation complete, waiting for output request
///                             All MAC results written to psum SPad
///                             Disable iact/wght SPad reads (computation finished)
///                             Setup psum SPad for sequential readout
///                             Reset psum address pointers to 0
///                             Waiting for external signal (psum_enable_i)
///                             Duration: Variable (waits for psum_enable_i)
///                             Transitions: psum_enable_i high → SEND_PSUM
///
///    State 8: SEND_PSUM     - Streaming partial sum results out
///                             Sequential read from psum SPad (addr 0 → PSUM_ADDR-1)
///                             Valid-ready handshake: psum_enable_o + psum_ready_i
///                             Increment psum address each successful transfer
///                             Data flows to downstream PE via psum_data_o
///                             Backpressure: Stalls if psum_ready_i low
///                             Duration: Variable (until all psums sent or psum_enable_i deasserts)
///                             Transitions: psum_enable_i low → IDLE (restart for next layer)
///
/// Pipeline Hazards and Bypass Logic:
///    Read-After-Write (RAW): When psum_spad read address == recent write address
///       Detection: Compare psum_spad_addr_a_r with psum_spad_addr_a_delay
///       Resolution: Set reuse_psum_spad_a flag, forward reused_data_a instead of SPad output
///       Mechanism: Bypass freshly computed psum from adder output to adder input
///    Adder-to-Adder Forwarding: Chain multiple accumulations
///       reuse_adder_data_a2a: Adder 1 output → Adder 1 input (serial mode)
///       reuse_adder_data_a2b: Adder 1 output → Adder 2 input (parallel mode)
///       reuse_adder_data_b2a: Adder 2 output → Adder 1 input (cross-lane)
///       reuse_adder_data_b2b: Adder 2 output → Adder 2 input (serial mode)
///

module PE #(

    parameter IS_TOPLEVEL = 1,
    parameter SERIAL      = 1,

    parameter integer PARALLEL_MACS = 2,

    parameter integer SPARSITY_EN      = 1,  // 1=sparse mode (default), 0=dense mode
    parameter integer USE_DSP          = 0,  // 0=standard multiplier+adder (default), 1=DSP48 slice optimization
    // Approach 3: when 1, expose horizontal iact pass-through ports for systolic GEMM dataflow
    parameter integer SYSTOLIC_GEMM_EN = 0,

    parameter integer DATA_IACT_BITWIDTH     = 8,
    parameter integer DATA_WGHT_BITWIDTH     = 8,
    parameter integer DATA_PSUM_BITWIDTH     = 20,
    parameter integer DATA_IACT_OVERHEAD     = 4,
    parameter integer DATA_WGHT_IGNORE_ZEROS = 4,

    parameter integer IACT_DATA_ADDR = 16,
    parameter integer IACT_ADDR_ADDR = 9,

    parameter integer WGHT_DATA_ADDR = 96,
    parameter integer WGHT_ADDR_ADDR = 16,

    parameter integer PSUM_ADDR = 32,

    parameter integer TRANS_BITWIDTH_IACT     = 24, // 3 * 8 bit data OR 2 * 12 bit data OR 6 * 4 bit addresses
    parameter integer TRANS_BITWIDTH_WGHT     = 24, // 3 * 8 bit weight OR 2 * 12 bit weight OR 3 * 8 bit addresses

    parameter integer NUM_GLB_IACT = 1,

    // local parameters
    localparam integer IACT_ADDR_DATA = $clog2(IACT_DATA_ADDR),
    localparam integer WGHT_ADDR_DATA = $clog2(WGHT_DATA_ADDR),

    localparam integer IACT_ADDR_ADDR_BITWIDTH = $clog2(IACT_ADDR_ADDR),

    localparam integer IACT_DATA_DATA          = SPARSITY_EN == 1 ? (DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD) : DATA_IACT_BITWIDTH,
    localparam integer IACT_DATA_ADDR_BITWIDTH = $clog2(IACT_DATA_ADDR),

    localparam integer WGHT_DATA_DATA          = SPARSITY_EN == 1 ? ((DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS) : (DATA_WGHT_BITWIDTH * PARALLEL_MACS),
    localparam integer WGHT_ADDR_ADDR_BITWIDTH = $clog2(WGHT_ADDR_ADDR),

    localparam integer WGHT_DATA_ADDR_BITWIDTH = $clog2(WGHT_DATA_ADDR),

    localparam integer PSUM_DATA = DATA_PSUM_BITWIDTH,
    localparam integer PSUM_ADDR_BITWIDTH = $clog2(PSUM_ADDR),
    localparam integer PSUM_WORDS_PER_TRANSFER = (SERIAL == 1 ? 1 : PARALLEL_MACS),
    localparam integer TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER

) (
    input                                             clk_i,
    input                                             rst_ni,
    input      [          $clog2(NUM_GLB_IACT+1)-1:0] iact_select_i,
    input      [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i,
    input      [                    NUM_GLB_IACT-1:0] iact_enable_i,
    output     [                    NUM_GLB_IACT-1:0] iact_ready_o,
    input      [             TRANS_BITWIDTH_WGHT-1:0] wght_data_i,
    input                                             wght_enable_i,
    output reg                                        wght_ready_o,
    input      [             TRANS_BITWIDTH_PSUM-1:0] psum_data_i,
    input                                             psum_enable_i,
    output                                            psum_ready_o,
    output     [             TRANS_BITWIDTH_PSUM-1:0] psum_data_o,
    output reg                                        psum_enable_o,
    input                                             psum_ready_i,
    input                                             compute_i,
    input                                             enable_stream_i,
    input      [                                11:0] data_stream_i,
    // Approach 3: horizontal iact pass-through for systolic GEMM
    // Active only when SYSTOLIC_GEMM_EN=1; otherwise tied to 0 / ignored
    input      [         DATA_IACT_BITWIDTH-1:0]      iact_pass_data_i,
    input                                             iact_pass_enable_i,
    output                                            iact_pass_ready_o,
    output     [         DATA_IACT_BITWIDTH-1:0]      iact_pass_data_o,
    output                                            iact_pass_enable_o,
    input                                             iact_pass_ready_i
);

  // ============================================================================
  // Internal Signal Declarations
  // ============================================================================

  // FSM state register for computation control
  reg  [                $clog2(16)-1:0] current_state_computing;

  // Scratch pad memory read data outputs
  wire  [          IACT_ADDR_DATA-1 : 0] iact_addr_SPad_data_r;  // Input activation address data
  wire  [          IACT_DATA_DATA-1 : 0] iact_data_SPad_data_r;  // Input activation payload data
  wire  [          WGHT_ADDR_DATA-1 : 0] wght_addr_SPad_data_r;  // Weight address data
  wire  [          WGHT_DATA_DATA-1 : 0] wght_data_SPad_data_r;  // Weight payload data

  // Scratch pad memory address registers
  reg  [   IACT_ADDR_ADDR_BITWIDTH-1:0] iact_addr_SPad_addr;    // Address for iact address SPad
  reg  [   IACT_DATA_ADDR_BITWIDTH-1:0] iact_data_SPad_addr;    // Address for iact data SPad
  wire [   WGHT_ADDR_ADDR_BITWIDTH-1:0] wght_addr_SPad_addr;    // Address for weight address SPad
  wire [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_SPad_addr;    // Address for weight data SPad

  // Data unpacking wires (separating payload from overhead/sparsity bits)
  wire [        DATA_IACT_BITWIDTH-1:0] iact_data_spad_pay;     // Iact payload (actual data value)
  wire [        DATA_IACT_OVERHEAD-1:0] iact_data_spad_oh;      // Iact overhead (tied to 0 in dense mode)

  // Weight data unpacking (for dual parallel MACs)
  wire [        DATA_WGHT_BITWIDTH-1:0] wght_data_spad_pay_1;   // Weight payload MAC 1
  wire [        DATA_WGHT_BITWIDTH-1:0] wght_data_spad_pay_2;   // Weight payload MAC 2
  wire [    DATA_WGHT_IGNORE_ZEROS-1:0] wght_data_spad_oh_1;    // Weight sparsity info MAC 1 (0 in dense)
  wire [    DATA_WGHT_IGNORE_ZEROS-1:0] wght_data_spad_oh_2;    // Weight sparsity info MAC 2 (0 in dense)

  // Partial sum scratch pad control signals
  reg                                   psum_data_SPad_en_a_r;  // Read enable for port A
  reg                                   psum_data_SPad_en_b_r;  // Read enable for port B
  reg                                   psum_data_SPad_en_a_w;  // Write enable for port A
  reg                                   psum_data_SPad_en_b_w;  // Write enable for port B

  // Partial sum data path control
  reg                                   psum_select;            // Mux select: 0=SPad data, 1=external psum
  reg                                   psum_enable;            // Psum valid signal (delayed)
  reg                                   psum_enable_2;          // Second delay stage for serial mode
  reg  [     TRANS_BITWIDTH_PSUM-1 : 0] psum_data_1_delay;     // Delayed psum input data 1
  reg  [     TRANS_BITWIDTH_PSUM-1 : 0] psum_data_2_delay;     // Delayed psum input data 2
  wire [   2*TRANS_BITWIDTH_PSUM-1 : 0] psum_data_combined_w;  // Combined psum for parallel mode

  // Control and handshaking signals
  reg                                   mux_iact_ready;         // Ready signal for iact multiplexer
  reg                                   adder_1_en;             // Enable signal for adder 1
  reg                                   adder_2_en;             // Enable signal for adder 2
  reg                                   adder_3_en;             // Enable signal for adder 3 (serial mode)

  // FSM control flags
  reg                                   fast_cycle;             // Flag for fast processing cycle
  reg                                   computing;              // Flag indicating active computation
  reg                                   iact_set;               // Flag: input activation data loaded
  reg                                   wght_set;               // Flag: weight data loaded
  wire                                  data_set;               // Both iact and wght data ready

  // Input activation multiplexer signals
  wire  [       TRANS_BITWIDTH_IACT-1:0] mux_iact_a_o_w;       // Mux data output (selected iact data)
  wire                                   mux_iact_b_o_w;        // Mux enable output (selected enable)
  wire                                   mux_iact_c_i_w;        // Mux ready input

  // Configuration and addressing registers
  reg  [          IACT_ADDR_DATA-1 : 0] iact_addr_max_reg;     // Max number of iact addresses

  // Pipeline registers for input activation data (3-stage delay line)
  reg  [      DATA_IACT_BITWIDTH-1 : 0] iact_data_current_3;   // Pipeline stage 3 (feeds multipliers)

  // Scratch pad read enable signals
  reg                                   iact_addr_SPad_en_r;    // Iact address SPad read enable
  reg                                   iact_data_SPad_en_r;    // Iact data SPad read enable
  reg                                   wght_addr_SPad_en_r;    // Weight address SPad read enable
  reg                                   wght_data_SPad_en_r;    // Weight data SPad read enable

  // Partial sum scratch pad addressing
  wire [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_a_r;    // Read address for port A
  wire [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_b_r;    // Read address for port B
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_a_mem;  // Memory base address for port A
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_b_mem;  // Memory base address for port B
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_a_delay;// Delayed write addr port A (for bypass)
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_b_delay;// Delayed write addr port B (for bypass)

  // Multiplier input signals
  wire [      DATA_WGHT_BITWIDTH-1 : 0] mult_1_fac_1;          // Multiplier 1 factor 1 (weight)
  wire [      DATA_IACT_BITWIDTH-1 : 0] mult_1_fac_2;          // Multiplier 1 factor 2 (iact)
  wire [      DATA_WGHT_BITWIDTH-1 : 0] mult_2_fac_1;          // Multiplier 2 factor 1 (weight)
  wire [      DATA_IACT_BITWIDTH-1 : 0] mult_2_fac_2;          // Multiplier 2 factor 2 (iact)

  // Partial sum memory usage tracking (bitmaps to track initialized locations)
  reg  [                 PSUM_ADDR-1:0] used_psum_memory;      // Parallel mode usage bitmap
  reg  [                 PSUM_ADDR-1:0] used_psum_memory_1;    // Serial mode usage bitmap MAC 1
  reg  [                 PSUM_ADDR-1:0] used_psum_memory_2;    // Serial mode usage bitmap MAC 2
  reg                                   use_psum_1;            // Flag: accumulate with existing psum 1
  reg                                   use_psum_2;            // Flag: accumulate with existing psum 2

  // Partial sum scratch pad data outputs
  wire [               PSUM_DATA-1 : 0] psum_spad_data_a_o;   // Data read from port A
  wire [               PSUM_DATA-1 : 0] psum_spad_data_b_o;   // Data read from port B
  // Adder input signals (summands)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_1_summand_1;   // Adder 1 input 1 (psum or 0)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_1_summand_2;   // Adder 1 input 2 (mult result)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_2_summand_1;   // Adder 2 input 1 (psum or 0)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_2_summand_2;   // Adder 2 input 2 (mult result)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_3_summand_1;   // Adder 3 input 1 (serial mode)
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_3_summand_2;   // Adder 3 input 2 (serial mode)

  // Computational unit outputs
  wire [      DATA_PSUM_BITWIDTH-1 : 0] mult_1_o_w;          // Multiplier 1 product output
  wire [      DATA_PSUM_BITWIDTH-1 : 0] mult_2_o_w;          // Multiplier 2 product output
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_1_o_w;         // Adder 1 sum output
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_2_o_w;         // Adder 2 sum output
  wire [      DATA_PSUM_BITWIDTH-1 : 0] adder_3_o_w;         // Adder 3 sum output (serial mode)

  // Data pipeline outputs for input activation (from data_pipeline_iact module)
  wire [ IACT_ADDR_ADDR_BITWIDTH-1 : 0] first_spad_iact_addr_w; // Write addr for iact addr SPad
  wire [          IACT_ADDR_DATA-1 : 0] first_spad_iact_data_w; // Write data for iact addr SPad
  wire                                  first_spad_iact_en_w;    // Write enable for iact addr SPad
  wire [ IACT_DATA_ADDR_BITWIDTH-1 : 0] second_spad_iact_addr_w;// Write addr for iact data SPad
  wire [          IACT_DATA_DATA-1 : 0] second_spad_iact_data_w;// Write data for iact data SPad
  wire                                  second_spad_iact_en_w;   // Write enable for iact data SPad

  // Data pipeline outputs for weights (from data_pipeline_wght module)
  wire [ WGHT_ADDR_ADDR_BITWIDTH-1 : 0] first_spad_wght_addr_w; // Write addr for wght addr SPad
  wire [          WGHT_ADDR_DATA-1 : 0] first_spad_wght_data_w; // Write data for wght addr SPad
  wire                                  first_spad_wght_en_w;    // Write enable for wght addr SPad
  wire [ WGHT_DATA_ADDR_BITWIDTH-1 : 0] second_spad_wght_addr_w;// Write addr for wght data SPad
  wire [          WGHT_DATA_DATA-1 : 0] second_spad_wght_data_w;// Write data for wght data SPad
  wire                                  second_spad_wght_en_w;   // Write enable for wght data SPad

  // Partial sum scratch pad write interface
  wire [      DATA_PSUM_BITWIDTH-1 : 0] psum_spad_data_a_i;    // Write data for port A
  wire [      DATA_PSUM_BITWIDTH-1 : 0] psum_spad_data_b_i;    // Write data for port B
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_a_w;    // Write address for port A
  reg  [      PSUM_ADDR_BITWIDTH-1 : 0] psum_spad_addr_b_w;    // Write address for port B

  // Data forwarding/bypass logic (for read-after-write hazard handling)
  reg                                   reuse_psum_spad_a;      // Bypass flag for port A
  reg                                   reuse_psum_spad_b;      // Bypass flag for port B
  reg  [      DATA_PSUM_BITWIDTH-1 : 0] reused_data_a;         // Bypassed data for port A
  reg  [      DATA_PSUM_BITWIDTH-1 : 0] reused_data_b;         // Bypassed data for port B
  wire                                  reuse_adder_data_a2a;   // Adder 1 output → Adder 1 input
  wire                                  reuse_adder_data_a2b;   // Adder 1 output → Adder 2 input
  wire                                  reuse_adder_data_b2a;   // Adder 2 output → Adder 1 input
  wire                                  reuse_adder_data_b2b;   // Adder 2 output → Adder 2 input

  // Weight data address register (used in both sparse and dense modes)
  reg  [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_vec;         // Weight data SPad address

  // Word counters from data pipeline modules (indicating amount of valid data loaded)
  wire [                         3 : 0] first_spad_words_iact_S;// # of words in iact addr SPad, in Eyeriss-Paper referenced as S
  wire [                         4 : 0] second_spad_words_iact; // # of words in iact data SPad
  wire [                         4 : 0] first_spad_words_wght;  // # of words in wght addr SPad
  wire [                         6 : 0] second_spad_words_wght; // # of words in wght data SPad

  // Computation control and configuration registers
  // [SPARSITY_EN=1 only] Weight data validity flag (always true in dense mode)
  reg                                   values_valid;           // Flag: current values are valid (not zero)
  reg  [                         4 : 0] filters_reg_M0;            // Number of filters configured, in Eyeriss-Paper referenced as M0
  reg  [                         3 : 0] channel_reg_C0;            // Number of channels configured, in Eyeriss-Paper referenced as C0
  wire                                  psum_data_SPad_en_a_w_i;// Internal write enable port A
  wire                                  psum_data_SPad_en_b_w_i;// Internal write enable port B
  reg                                   raw_wght_reg;           // 1 = raw (uncompressed) weight stream: keep all-zero weight words
  reg  [                           3:0] iact_x_line_repetitions;

  // Configuration streaming FSM
  reg  [                           1:0] current_state_stream;   // Config stream state
  // Approach 3: systolic pass-through (reg when SYSTOLIC_GEMM_EN=1, wire otherwise)
  reg  [         DATA_IACT_BITWIDTH-1:0] iact_pass_data_reg;    // Registered iact value to forward
  reg                                    iact_pass_enable_reg;   // Registered enable to forward

  // Output formatting
  wire [        DATA_PSUM_BITWIDTH-1:0] output_adder;          // Combined output from both adders

  // ============================================================================
  // FST Waveform Dump Configuration (for CocoTB simulation)
  // ============================================================================
`ifndef NO_TRACE
  reg[1023:0] fst_path;
  initial begin
    if (IS_TOPLEVEL) begin
      // Read the path from the command line argument
      if ($value$plusargs("FST_PATH=%s", fst_path)) begin
        $dumpfile(fst_path);
        $dumpvars(0, PE);
      end else begin
        // Fallback for when the argument is not provided
        $dumpfile("PE.fst");
        $dumpvars(0, PE);
      end
    end
  end
`endif

  // ============================================================================
  // Approach 3: Systolic iact pass-through logic
  // ============================================================================
  // When SYSTOLIC_GEMM_EN=1, each PE registers the incoming iact value and
  // forwards it one cycle later to the next PE in the same row.  The PE also
  // captures iact_pass_data_i into iact_data_current_3 so the standard MAC
  // pipeline uses the streamed value rather than the local SPad.
  generate
    if (SYSTOLIC_GEMM_EN) begin : gen_systolic_pass
      // One pipeline register: accept → hold for one cycle → forward
      always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          iact_pass_data_reg   <= {DATA_IACT_BITWIDTH{1'b0}};
          iact_pass_enable_reg <= 1'b0;
        end else begin
          iact_pass_data_reg   <= iact_pass_data_i;
          iact_pass_enable_reg <= iact_pass_enable_i;
        end
      end
      // Outputs: forward delayed value downstream
      assign iact_pass_data_o   = iact_pass_data_reg;
      assign iact_pass_enable_o = iact_pass_enable_reg;
      // Always ready to accept (single-register, no backpressure in this dataflow)
      assign iact_pass_ready_o  = 1'b1;
    end else begin : gen_no_systolic_pass
      // Tie off all outputs when feature is disabled; drive regs to avoid X
      initial begin
        iact_pass_data_reg   = {DATA_IACT_BITWIDTH{1'b0}};
        iact_pass_enable_reg = 1'b0;
      end
      assign iact_pass_data_o   = {DATA_IACT_BITWIDTH{1'b0}};
      assign iact_pass_enable_o = 1'b0;
      assign iact_pass_ready_o  = 1'b0;
    end
  endgenerate

  // ============================================================================
  // FSM State Definitions
  // ============================================================================

  // Configuration streaming FSM states (for receiving configuration parameters)
  localparam [1:0] FIRST_PARAMS  = 0;  // Receive stride, weight addr max, data mode
  localparam [1:0] SECOND_PARAMS = 1;  // Receive filter count, channel count
  localparam [1:0] THIRD_PARAMS  = 2;  // Receive iact addr max
  localparam [1:0] FOURTH_PARAMS = 3;  // Final parameter state

  // Main computation FSM states
  // IDLE:              Waiting for data to be loaded, ready to accept new computation
  // LOADING_1-5:       Preparation phases - loading and prefetching data from SPads
  // CALCULATING:       Active computation using MAC operations
  // WAIT_TO_SEND_PSUM: Waiting for external ready signal to send partial sums out
  // SEND_PSUM:         Streaming partial sum results to next stage
  localparam [$clog2(16)-1:0] IDLE              = 0;
  localparam [$clog2(16)-1:0] LOADING_1         = 1;
  localparam [$clog2(16)-1:0] LOADING_2         = 2;
  localparam [$clog2(16)-1:0] LOADING_3         = 3;
  localparam [$clog2(16)-1:0] LOADING_4         = 4;
  localparam [$clog2(16)-1:0] LOADING_5         = 5;
  localparam [$clog2(16)-1:0] CALCULATING       = 6;
  localparam [$clog2(16)-1:0] WAIT_TO_SEND_PSUM = 7;
  localparam [$clog2(16)-1:0] SEND_PSUM         = 8;

  // ============================================================================
  // Combinational Logic Assignments
  // ============================================================================

  // Data loading handshaking - both iact and wght must be loaded before compute
  assign data_set       = iact_set & wght_set;
  assign mux_iact_c_i_w = mux_iact_ready;

  // Unpack iact data SPad output: conditional based on SPARSITY_EN
  generate
    if (SPARSITY_EN == 1) begin : gen_sparse_iact_unpack
      // Sparse mode: extract overhead bits from packed data
      assign {iact_data_spad_oh, iact_data_spad_pay} = iact_data_SPad_data_r;
    end else begin : gen_dense_iact_unpack
      // Dense mode: no overhead, full word is payload
      assign iact_data_spad_oh = {DATA_IACT_OVERHEAD{1'b0}};
      assign iact_data_spad_pay = iact_data_SPad_data_r[DATA_IACT_BITWIDTH-1:0];
    end
  endgenerate

  // Unpack weight data SPad output: conditional based on SPARSITY_EN
  generate
    if (SPARSITY_EN == 1) begin : gen_sparse_wght_unpack
      // Sparse mode: extract overhead bits for dual MACs
      assign {wght_data_spad_oh_2, wght_data_spad_pay_2, wght_data_spad_oh_1, wght_data_spad_pay_1} = wght_data_SPad_data_r;
    end else begin : gen_dense_wght_unpack
      // Dense mode: no overhead, just two weight values
      assign wght_data_spad_oh_1 = {DATA_WGHT_IGNORE_ZEROS{1'b0}};
      assign wght_data_spad_oh_2 = {DATA_WGHT_IGNORE_ZEROS{1'b0}};
      assign wght_data_spad_pay_1 = wght_data_SPad_data_r[DATA_WGHT_BITWIDTH-1:0];
      assign wght_data_spad_pay_2 = wght_data_SPad_data_r[2*DATA_WGHT_BITWIDTH-1:DATA_WGHT_BITWIDTH];
    end
  endgenerate

  // Adder 3 inputs (serial mode only - combines outputs of adder 1 and 2)
  assign adder_3_summand_1 = SERIAL == 1 ? adder_1_o_w : 0;
  assign adder_3_summand_2 = SERIAL == 1 ? adder_2_o_w : 0;

  // Output mux: serial mode outputs single psum, parallel mode outputs combined
  assign psum_data_o = SERIAL == 1 ? {{(TRANS_BITWIDTH_PSUM-DATA_PSUM_BITWIDTH){1'd0}}, adder_3_o_w} : output_adder[TRANS_BITWIDTH_PSUM-1:0];
  assign output_adder = adder_1_o_w;

  // Weight address generation:
  // SPARSITY_EN=1: use vector or compute from iact overhead (for zero-skipping)
  // SPARSITY_EN=0: weight addr SPad unused; wght_data_SPad_addr driven by dense FSM
  generate
    if (SPARSITY_EN == 1) begin : gen_sparse_wght_addr
      assign wght_addr_SPad_addr = gen_sparse_fsm.wght_addr_use_vec ? gen_sparse_fsm.wght_addr_vec : (iact_data_spad_oh == 0 ? 0 : (iact_data_spad_oh - 1));
      assign wght_data_SPad_addr = gen_sparse_fsm.wght_data_use_vec ? wght_data_vec : wght_addr_SPad_data_r;
    end else begin : gen_dense_wght_addr
      assign wght_addr_SPad_addr = {WGHT_ADDR_ADDR_BITWIDTH{1'b0}}; // Unused in dense mode
      assign wght_data_SPad_addr = wght_data_vec;                    // Directly driven by dense FSM
    end
  endgenerate

  // Multiplier inputs: weights go to factor 1, iact goes to factor 2
  // Approach 3: in systolic mode use the pass-through value instead of the SPad pipeline
  assign mult_1_fac_1 = wght_data_spad_pay_1;
  assign mult_2_fac_1 = wght_data_spad_pay_2;
  // Approach 3: in systolic mode use the pass-through value instead of the SPad pipeline
  assign mult_1_fac_2 = (SYSTOLIC_GEMM_EN && iact_pass_enable_i) ? iact_pass_data_i : iact_data_current_3;
  assign mult_2_fac_2 = (SYSTOLIC_GEMM_EN && iact_pass_enable_i) ? iact_pass_data_i : iact_data_current_3;

  // Data forwarding/bypass detection logic (detects read-after-write hazards)
  // These signals indicate when the data being read is the same location just written
  assign reuse_adder_data_a2a = (psum_spad_addr_a_delay == psum_spad_addr_a_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_a2b = SERIAL ? 0: (psum_spad_addr_b_delay == psum_spad_addr_a_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_b2a = SERIAL ? 0: (psum_spad_addr_a_delay == psum_spad_addr_b_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_b2b = (psum_spad_addr_b_delay == psum_spad_addr_b_w) & (current_state_computing != SEND_PSUM);

  // Adder 1 summand 1: select psum source with bypass logic
  // Priority: zero if not accumulating > adder bypass > SPad bypass > normal SPad read
  assign adder_1_summand_1 =
                 !use_psum_1 ? 0 :                      // First accumulation: use 0
                (reuse_adder_data_a2a ? adder_1_o_w :   // Forward adder 1 output
                (reuse_adder_data_b2a ? adder_2_o_w :   // Forward adder 2 output
                (reuse_psum_spad_a ? reused_data_a :    // Use bypassed SPad data
                 psum_spad_data_a_o)));                 // Normal SPad read

  // Adder 2 summand 1: select psum source with bypass logic (same structure as adder 1)
  assign adder_2_summand_1 =
                 !use_psum_2 ? 0 :
                (reuse_adder_data_b2b ? adder_2_o_w :
                (reuse_adder_data_a2b ? adder_1_o_w :
                (reuse_psum_spad_b ? reused_data_b :
                 psum_spad_data_b_o)));

  // Write data to psum SPad comes from adder outputs
  assign psum_spad_data_a_i = adder_1_o_w;
  assign psum_spad_data_b_i = adder_2_o_w;

  // Combine delayed psum inputs for parallel mode
  assign psum_data_combined_w = {psum_data_2_delay, psum_data_1_delay};

  // Psum SPad read address calculation
  // During output: use base address directly
  // During compute:
  //   - SPARSITY_EN=1: add weight sparsity offset to base address (for sparse indexing)
  //   - SPARSITY_EN=0: use base address directly (no sparse offset)
  generate
    if (SPARSITY_EN == 1) begin : gen_sparse_psum_addr
      assign psum_spad_addr_a_r = ((current_state_computing == WAIT_TO_SEND_PSUM) | (current_state_computing == SEND_PSUM)) ?
                                    psum_spad_addr_a_mem :
                                    wght_data_spad_oh_1 + psum_spad_addr_a_mem;
      assign psum_spad_addr_b_r = ((current_state_computing == WAIT_TO_SEND_PSUM) | (current_state_computing == SEND_PSUM)) ?
                                    psum_spad_addr_b_mem :
                                    wght_data_spad_oh_1 + wght_data_spad_oh_2 + psum_spad_addr_b_mem;
    end else begin : gen_dense_psum_addr
      // Dense mode: no sparsity offset, use base address directly
      assign psum_spad_addr_a_r = psum_spad_addr_a_mem;
      assign psum_spad_addr_b_r = psum_spad_addr_b_mem;
    end
  endgenerate

  // Psum SPad write enable logic (prevent write conflicts when addresses match)
  assign psum_data_SPad_en_a_w_i = !psum_data_SPad_en_a_w ? 0 :
                                      !psum_data_SPad_en_a_r ? 1 :
                                       (psum_spad_addr_a_w != psum_spad_addr_a_r) ?  1 : 0;
  assign psum_data_SPad_en_b_w_i = !psum_data_SPad_en_b_w ? 0 :
                                      !psum_data_SPad_en_b_r ? 1 :
                                       (psum_spad_addr_b_w != psum_spad_addr_b_r) ?  1 : 0;

  // Psum output ready signal (gated by internal select)
  assign psum_ready_o = psum_ready_i & psum_select;

  // Calculated ceiled filters from filters depending on PARALLEL_MACS

  // ============================================================================
  // Configuration Parameter Streaming FSM
  // ============================================================================
  // This FSM receives configuration parameters via the data_stream_i interface
  // Parameters are received in four sequential states and stored in registers
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      raw_wght_reg            <= 0;
      current_state_stream    <= 0;
      iact_addr_max_reg       <= 0;
      iact_x_line_repetitions <= 0;
      filters_reg_M0          <= 0;
      channel_reg_C0          <= 0;
    end else begin
      case (current_state_stream)
        FIRST_PARAMS: begin
          // Receive first set of parameters: stride, weight address max
          if (enable_stream_i) begin
            current_state_stream  <= SECOND_PARAMS;
            raw_wght_reg            <= data_stream_i[9];      // Raw (uncompressed) weight stream flag
          end
        end
        SECOND_PARAMS: begin
          // Receive second set: filter count, channel count
          if (enable_stream_i) begin
            current_state_stream <= THIRD_PARAMS;
            filters_reg_M0          <= data_stream_i[8:4];     // Number of filters
            channel_reg_C0          <= data_stream_i[3:0];     // Number of channels
          end else begin
            current_state_stream <= FIRST_PARAMS;           // Timeout: restart
          end
        end
        THIRD_PARAMS: begin
          // Receive third set: input activation address max
          if (enable_stream_i) begin
            iact_addr_max_reg       <= data_stream_i[3:0];     // Max iact addresses
            iact_x_line_repetitions <= data_stream_i[7:4];
            current_state_stream    <= FOURTH_PARAMS;
          end else begin
            current_state_stream <= FIRST_PARAMS;           // Timeout: restart
          end
        end
        FOURTH_PARAMS: begin
          // Fourth parameter state (currently unused, returns to FIRST)
          if (enable_stream_i) begin
            current_state_stream <= FIRST_PARAMS;
          end else begin
            current_state_stream <= FIRST_PARAMS;
          end
        end
        default: begin
        end
      endcase
    end
  end

  // ============================================================================
  // Data Loading Handshake Flags
  // ============================================================================
  // Tracks when input activation and weight data have been loaded into SPads
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      iact_set <= 0;
      wght_set <= 0;
    end else begin
      if (enable_stream_i) begin
        // Reset flags when new configuration is streaming
        iact_set <= 0;
        wght_set <= 0;
      end else begin
        // Set flag when iact data arrives
        if (mux_iact_b_o_w) begin
          iact_set <= 1;
        end
        // Set flag when weight data arrives
        if (wght_enable_i) begin
          wght_set <= 1;
        end
      end
    end
  end

  // ============================================================================
  // Main Computation Control FSM
  // ============================================================================
  // Two alternative implementations selected by the SPARSITY_EN parameter:
  //   SPARSITY_EN=1: Full sparse FSM with zero-skipping, overhead-bit pipelines,
  //                  weight address range computation, and activation advance logic.
  //   SPARSITY_EN=0: Simplified dense FSM with sequential weight/iact addressing,
  //                  no zero-skipping, no overhead bits, no weight range computation.
    generate if (SPARSITY_EN == 1) begin : gen_sparse_fsm
    // --------------------------------------------------------------------------
    // Sparsity-only registers (only exist when SPARSITY_EN=1)
    // --------------------------------------------------------------------------
    reg  [      DATA_IACT_BITWIDTH-1 : 0] iact_data_current_1;   // Pipeline stage 1
    reg  [      DATA_IACT_BITWIDTH-1 : 0] iact_data_current_2;   // Pipeline stage 2
    reg  [        DATA_IACT_OVERHEAD-1:0] iact_oh_delay_1;       // Pipeline delay stage 1 for overhead
    reg  [        DATA_IACT_OVERHEAD-1:0] iact_oh_delay_2;       // Pipeline delay stage 2 for overhead
    reg                                   next_iact;             // Flag to advance to next input activation
    reg  [          IACT_ADDR_DATA-1 : 0] iact_addr_current;     // Current iact address being processed
    reg  [          IACT_ADDR_DATA-1 : 0] iact_addr_count;       // Counter for iact addresses processed
    reg                                   wght_addr_use_vec;     // Mux select: use vector addr or computed
    reg  [   WGHT_ADDR_ADDR_BITWIDTH-1:0] wght_addr_vec;         // Vector-based weight address
    reg                                   wght_data_use_vec;     // Mux select: use vector data or computed
    reg  [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_start;       // Start address for weight data range
    reg  [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_end;         // End address for weight data range
    reg  [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_end_pre;     // Pre-computed end for next range
    reg  [   WGHT_DATA_ADDR_BITWIDTH-1:0] wght_data_start_pre;   // Pre-computed start for next range
    reg                                   wght_start_set;        // Flag: start address has been set
    reg                                   wght_end_set;          // Flag: end address has been set

    // ============================================================================
    // SPARSE FSM (SPARSITY_EN=1)
    // ============================================================================
    // This is the primary FSM controlling PE operation through multiple phases:
    // - Data loading and preparation (IDLE, LOADING states)
    // - MAC computation with zero-skipping (CALCULATING state)
    // - Partial sum output (WAIT_TO_SEND_PSUM, SEND_PSUM states)
    always @(posedge clk_i, negedge rst_ni) begin
      if (!rst_ni) begin
        current_state_computing <= IDLE;
        iact_addr_SPad_addr     <= 0;
        iact_addr_SPad_en_r     <= 0;
        iact_addr_current       <= 0;
        iact_addr_count         <= 0;
        iact_data_SPad_addr     <= 0;
        iact_data_SPad_en_r     <= 0;
        wght_addr_SPad_en_r     <= 0;
        wght_addr_use_vec       <= 1;
        wght_data_use_vec       <= 1;
        wght_addr_vec           <= 0;
        wght_data_vec           <= 0;
        wght_data_start         <= 0;
        wght_start_set          <= 0;
        wght_data_end           <= 0;
        wght_data_end_pre       <= 0;
        wght_data_start_pre     <= 0;
        wght_end_set            <= 0;
        wght_data_SPad_en_r     <= 0;
        psum_data_SPad_en_a_r   <= 0;
        psum_data_SPad_en_b_r   <= 0;
        psum_data_SPad_en_a_w   <= 0;
        psum_data_SPad_en_b_w   <= 0;
        iact_data_current_1     <= 0;
        iact_data_current_2     <= 0;
        iact_data_current_3     <= 0;
        iact_oh_delay_1         <= 0;
        iact_oh_delay_2         <= 0;
        computing               <= 0;
        fast_cycle              <= 0;
        next_iact               <= 0;
        used_psum_memory        <= 0;
        use_psum_1              <= 0;
        use_psum_2              <= 0;
        adder_1_en              <= 0;
        adder_2_en              <= 0;
        mux_iact_ready          <= 1;
        wght_ready_o            <= 1;
        psum_select             <= 1;
        psum_data_1_delay       <= 0;
        psum_data_2_delay       <= 0;
        psum_enable             <= 0;
        psum_enable_2           <= 0;
        psum_enable_o           <= 0;
        psum_spad_addr_a_mem    <= 0;
        psum_spad_addr_b_mem    <= 1;
        reuse_psum_spad_a       <= 0;
        reuse_psum_spad_b       <= 0;
        reused_data_a           <= 0;
        reused_data_b           <= 0;
        values_valid            <= 0;
        psum_spad_addr_a_delay  <= 0;
        psum_spad_addr_b_delay  <= 1;
        psum_spad_addr_a_w      <= 0;
        psum_spad_addr_b_w      <= 1;
      end else begin
        // Normal operation: Update internal states and handle partial sum (psum) pipeline
        if (psum_ready_i) begin
          psum_enable <= psum_enable_i;
        end else begin
          psum_enable <= 0;
        end
        {psum_data_2_delay, psum_data_1_delay} <= {{(TRANS_BITWIDTH_PSUM) {1'd0}}, psum_data_i};
        if (SERIAL == 1) begin
          psum_enable_2 <= psum_enable;
          psum_enable_o <= psum_enable_2;
        end else begin
          psum_enable_o <= psum_enable;
        end
        iact_oh_delay_1 <= iact_data_spad_oh == 0 ? 0 : iact_data_spad_oh - 1;
        iact_oh_delay_2 <= iact_oh_delay_1;
        case (current_state_computing)
          // ====================================================================
          // IDLE State: Waiting for computation trigger or psum output request
          // ====================================================================
          IDLE: begin
            // Reset control signals and prepare for next operation
            mux_iact_ready         <= 1;
            iact_addr_current      <= 0;
            iact_addr_count        <= 0;
            wght_ready_o           <= 1;
            wght_addr_vec          <= 0;
            wght_data_vec          <= 0;
            wght_data_start        <= 0;
            wght_start_set         <= 0;
            wght_data_end          <= 0;
            wght_data_end_pre      <= 0;
            wght_data_start_pre    <= 0;
            wght_end_set           <= 0;
            //Read first values
            fast_cycle             <= 0;
            next_iact              <= 0;
            iact_addr_SPad_addr    <= 0;
            iact_addr_SPad_en_r    <= 0;
            iact_data_SPad_addr    <= 0;
            iact_data_SPad_en_r    <= 0;
            iact_oh_delay_1        <= 0;
            iact_oh_delay_2        <= 0;
            wght_addr_SPad_en_r    <= 0;
            wght_addr_use_vec      <= 1;
            wght_data_use_vec      <= 1;
            wght_data_SPad_en_r    <= 0;
            psum_data_SPad_en_a_r  <= computing;
            psum_data_SPad_en_b_r  <= computing;
            psum_data_SPad_en_a_w  <= 0;
            psum_data_SPad_en_b_w  <= 0;
            computing              <= 0;
            values_valid           <= 0;
            psum_spad_addr_a_delay <= 0;
            psum_spad_addr_b_delay <= 1;
            psum_spad_addr_a_w     <= 0;
            psum_spad_addr_b_w     <= 1;
            psum_spad_addr_a_mem   <= 0;
            psum_spad_addr_b_mem   <= 1;
            adder_1_en             <= 0;
            adder_2_en             <= 0;
            adder_3_en             <= 0;
            reuse_psum_spad_a      <= 0;
            reuse_psum_spad_b      <= 0;
            reused_data_a          <= 0;
            reused_data_b          <= 0;
            use_psum_1             <= 0;
            use_psum_2             <= 0;
            psum_select            <= 1;
            if (SERIAL == 1) begin
              used_psum_memory_1 <= 0;
              used_psum_memory_2 <= 0;
            end else begin
              used_psum_memory <= 0;
            end
            // Check if external system wants to read partial sums
            if (psum_enable_i) begin
              current_state_computing <= SEND_PSUM;
              adder_1_en              <= 1;
              adder_2_en              <= 1;
              psum_select             <= 1;
              use_psum_1              <= 0;
              use_psum_2              <= 0;
              if (SERIAL == 1) begin
                used_psum_memory_1 <= 0;
                used_psum_memory_2 <= 0;
              end else begin
                used_psum_memory <= 0;
              end
            end
            // Check if ready to start computation (data loaded, compute trigger, valid data)
            if (data_set & compute_i & ((second_spad_words_iact != 0) & (second_spad_words_wght != 0))) begin
              // Initiate computation sequence
              current_state_computing <= LOADING_1;
              mux_iact_ready          <= 0;
              wght_ready_o            <= 0;
              psum_select             <= 0;
              iact_addr_SPad_en_r     <= 1;
              iact_data_SPad_addr     <= 0;
              iact_data_SPad_en_r     <= 1;
              psum_data_SPad_en_a_r   <= 0;
              psum_data_SPad_en_b_r   <= 0;
              use_psum_1              <= 0;
              use_psum_2              <= 0;
              if (SERIAL == 1) begin
                used_psum_memory_1 <= 0;
                used_psum_memory_2 <= 0;
              end else begin
                used_psum_memory <= 0;
              end
            end
          end

          // ====================================================================
          // LOADING_1: Fetch first weight address and iact data
          // ====================================================================
          // Rationale: Initiate the weight-activation loading pipeline
          // - Fetch the weight address from the weight address scratchpad (SPad)
          // - Advance to the next iact data entry in the iact SPad
          // - Setup the iact address from the max register for proper indexing
          // Signal assignments:
          //   wght_addr_use_vec <= 0: Don't use computed weight vector yet
          //   wght_addr_SPad_en_r <= 1: Enable reading weight addresses
          //   iact_data_SPad_addr + 1: Move to next iact entry
          //   iact_addr_SPad_addr <= max-1: Set iact address for current iteration
          //   iact_oh_delay_1 <= oh: Latch overhead bits for zero-skipping
          LOADING_1: begin
            current_state_computing <= LOADING_2;
            wght_addr_use_vec       <= 0;
            wght_addr_SPad_en_r     <= 1;
            iact_data_SPad_addr     <= iact_data_SPad_addr + 1;
            iact_addr_SPad_addr     <= iact_addr_max_reg - 1;
            iact_oh_delay_1         <= iact_data_spad_oh;
          end

          // ====================================================================
          // LOADING_2: Compute weight data address from weight addr SPad
          // ====================================================================
          // Rationale: Calculate the actual weight data address and latch iact
          // - Use the fetched weight address to locate weight data in SPad
          // - Latch the first iact data value from the previous cycle
          // - Store the iact address for use in MAC operations
          // - Disable weight address SPad enable (single cycle fetch)
          // Signal assignments:
          //   wght_data_SPad_en_r <= 1: Enable weight data SPad read
          //   wght_addr_use_vec <= 1: Use the fetched address for weight lookup
          //   wght_addr_vec <= address + 1: Pre-increment for next access
          //   iact_data_current_1 <= payload: Capture first iact value
          //   iact_addr_current <= fetched_addr: Store for MAC loop
          //   Special case: If overhead bits = 0, reset to address 1
          LOADING_2: begin
            current_state_computing <= LOADING_3;
            wght_data_SPad_en_r <= 1;
            wght_addr_use_vec   <= 1;
            wght_addr_vec       <= wght_addr_SPad_addr + 1;
            iact_data_current_1 <= iact_data_spad_pay;
            iact_addr_current   <= iact_addr_SPad_data_r;
            iact_addr_SPad_en_r <= 0;
            iact_oh_delay_1     <= iact_data_spad_oh;
            if (iact_data_spad_oh == 0) begin
              iact_data_SPad_addr <= 1;
              wght_addr_vec       <= 0;
            end
          end

          // ====================================================================
          // LOADING_3: Determine weight data start address for zero-skipping
          // ====================================================================
          // Rationale: Setup the weight range for sparse computation
          // - Determine the starting weight address based on overhead bits
          // - Handle the zero-skipping case where no activations exist (oh=0)
          // - Pipeline the iact data through delay registers for alignment
          // Signal assignments:
          //   wght_addr_use_vec <= 0: Stop using weight address vector
          //   wght_data_use_vec <= 0: Not yet using weight data vector
          //   iact_data_current_2 <= current_1: Shift iact pipeline
          //   iact_data_SPad_addr + 1: Advance to next iact entry
          //   wght_data_start <= fetched_addr: Set weight range start
          //   Zero-case: If oh=0, set start to 0, end to fetched address
          LOADING_3: begin
            current_state_computing <= LOADING_4;
            wght_addr_use_vec       <= 0;
            wght_data_use_vec       <= 0;
            iact_data_current_1     <= iact_data_spad_pay;
            iact_data_current_2     <= iact_data_current_1;
            iact_data_SPad_addr     <= iact_data_SPad_addr + 1;
            iact_addr_SPad_en_r     <= 0;
            wght_data_start         <= wght_addr_SPad_data_r;
            //Zero-Case
            if (iact_oh_delay_1 == 0) begin
              wght_data_start <= 0;
              wght_data_end   <= wght_addr_SPad_data_r;
            end
          end

          // ====================================================================
          // LOADING_4: Determine weight data end address for zero-skipping
          // ====================================================================
          // Rationale: Complete the weight range definition for the MAC loop
          // - Determine ending weight address to define the sparse computation range
          // - Start using weight data and addresses for MAC operations
          // - Continue iact data pipeline advancement
          // Signal assignments:
          //   iact_data_SPad_addr + 1: Continue advancing iact entries
          //   wght_addr_use_vec <= 1: Enable using computed weight addresses
          //   wght_data_use_vec <= 1: Enable weight data SPad for MAC
          //   wght_data_vec <= start_addr: Begin at calculated start address
          //   wght_addr_vec + 1: Pre-increment for next weight address
          //   Zero-case: If oh!=0, set end address from next fetched value
          LOADING_4: begin
            current_state_computing <= LOADING_5;
            iact_data_SPad_addr     <= iact_data_SPad_addr + 1;
            wght_addr_use_vec       <= 1;
            wght_data_use_vec       <= 1;
            wght_data_vec           <= wght_data_start;
            wght_addr_vec           <= wght_addr_SPad_addr + 1;
            iact_addr_SPad_en_r      <= 0;
            //Zero-Case
            if (iact_oh_delay_2 != 0) begin
              wght_data_end <= wght_addr_SPad_data_r;
            end
          end

          // ====================================================================
          // LOADING_5: Final preparation, fetch next weight range, enter compute
          // ====================================================================
          // Rationale: Complete initialization and transition to MAC computation
          // - Signal readiness to start multiply-accumulate operations
          // - Prepare the next weight range addresses for seamless iteration
          // - Finalize the iact data pipeline alignment (3-cycle delay established)
          // - Prepare for fast cycling through weight vectors
          // Signal assignments:
          //   current_state_computing <= CALCULATING: Transition to MAC state
          //   wght_start_set <= 1: Mark weight start address as valid
          //   computing <= 1: Enable MAC operations
          //   wght_addr_use_vec <= 1: Use weight addresses in MAC
          //   values_valid <= 1: Data is ready if weight range is valid (end > start)
          //   wght_data_vec + 1: Pre-increment for first MAC iteration
          //   iact_data_current_3 <= current_2: Complete 3-cycle pipeline delay
          //   fast_cycle <= 1: Enable fast cycling for consecutive activations
          //   Special cases: Handle iact_addr_current==1 and zero address scenarios
          LOADING_5: begin
            current_state_computing <= CALCULATING;
            wght_start_set          <= 1;
            computing               <= 1;
            wght_addr_use_vec       <= 1;
            if (wght_data_end > wght_data_start) begin
              values_valid <= 1;
            end
            wght_data_vec <= wght_data_vec + 1;
            wght_data_start_pre <= wght_addr_SPad_data_r;
            fast_cycle          <= 1;
            iact_addr_count     <= 1;
            iact_addr_SPad_en_r <= 0;
            if ((iact_addr_SPad_data_r == 0) & (iact_addr_current == 0)) begin
              iact_addr_SPad_en_r <= 1;
            end
            wght_addr_vec       <= iact_data_spad_oh - 1;
            iact_data_current_1 <= iact_data_spad_pay;
            iact_data_current_2 <= iact_data_current_1;
            iact_data_current_3 <= iact_data_current_2;
            iact_addr_current   <= 0;
          end

          // ====================================================================
          // CALCULATING: Main MAC computation loop with zero-skipping
          // ====================================================================
          // Performs multiply-accumulate operations using:
          // - Dual parallel multipliers (for two weight-iact pairs)
          // - Sparse data indexing (skipping zeros via overhead bits)
          // - Data forwarding to handle read-after-write hazards
          // - Psum memory usage tracking to determine accumulation vs. first write
          //
          // MAJOR CODE BLOCKS:
          // 1. DEFAULT SIGNAL SETUP: Initialize all control signals for normal operation
          // 2. WEIGHT RANGE COMPUTATION: Compute start/end addresses for each activation
          // 3. NEXT ACTIVATION TRIGGER: Determine when to move to next activation
          // 4. DATA VALIDITY CHECK: Verify current weight vector is in valid range
          // 5. NEXT ACTIVATION HANDLING: Update data pipeline when moving to next activation
          // 6. READ-AFTER-WRITE FORWARDING: Detect and forward recently computed psums
          // 7. PSUM MEMORY TRACKING: Track which psum locations have been written
          // 8. COMPLETION CHECK: Detect end of computation for this activation
          CALCULATING: begin
              // ================================================================
              // BLOCK 1: DEFAULT SIGNAL SETUP
              // ================================================================
              // Initialize control signals for normal MAC operation
              // Rationale: Set defaults for all SPad enables and control flags
              iact_addr_SPad_en_r   <= 0;            // Disable iact address reads
              iact_data_SPad_en_r   <= !mux_iact_ready; // Enable iact data when ready
              wght_addr_SPad_en_r   <= $bits(wght_addr_SPad_en_r)'(SPARSITY_EN);  // Enable weight address reads only in sparse mode
              psum_data_SPad_en_a_r <= computing;    // Read psum port A when computing
              psum_data_SPad_en_b_r <= computing;    // Read psum port B when computing
              psum_data_SPad_en_a_w <= 0;            // Disable psum writes (default)
              psum_data_SPad_en_b_w <= 0;            // Disable psum writes (default)
              reuse_psum_spad_a     <= 0;            // No data forwarding (default)
              reuse_psum_spad_b     <= 0;            // No data forwarding (default)
              reused_data_a         <= 0;            // No forwarded data (default)
              reused_data_b         <= 0;            // No forwarded data (default)
              wght_addr_use_vec     <= 1;            // Use computed weight addresses
              fast_cycle            <= 0;            // Not in fast cycling mode
              next_iact             <= 0;            // Don't advance activation yet
              psum_spad_addr_a_mem  <= psum_spad_addr_b_r + 1; // Next psum A memory
              psum_spad_addr_b_mem  <= psum_spad_addr_b_r + 2; // Next psum B memory

              // ================================================================
              // BLOCK 1b: WEIGHT VECTOR INCREMENT
              // ================================================================
              // Advance through weight data for current activation
              if (wght_data_vec < (second_spad_words_wght - 1)) begin
                wght_data_vec <= wght_data_vec + 1; // Increment weight index
              end else begin
                mux_iact_ready <= 1;                 // Signal ready for next activation
              end

              // ================================================================
              // BLOCK 2: WEIGHT RANGE COMPUTATION
              // ================================================================
              // Compute start/end addresses for current activation's weights
              // Rationale: Use overhead bits and weight address SPad to determine
              // the range of weights that correspond to non-zero activations
              if (!next_iact || fast_cycle) begin
                if (wght_start_set) begin
                  // Weight start address already set, now compute end address
                  if (!wght_end_set) begin
                    wght_data_end_pre <= wght_addr_SPad_data_r; // Fetch end address
                    wght_end_set      <= 1;                      // Mark end as valid
                  end
                end else begin
                  // First weight in this activation range
                  wght_data_start_pre <= wght_addr_SPad_data_r; // Fetch start address
                  wght_start_set <= 1;                           // Mark start as valid
                  // Adjust weight address vector based on overhead bits
                  if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                    wght_addr_vec <= iact_oh_delay_1; // Use overhead to index weights
                  end
                end
              end

              // ================================================================
              // BLOCK 3: NEXT ACTIVATION TRIGGER
              // ================================================================
              // Detect when weight range is completely fetched, trigger next activation
              // Rationale: When we've fetched all weight range data, prepare to advance
              // to the next activation in the input sequence
              if ((wght_data_end <= wght_data_SPad_addr + 1) && !next_iact) begin
                wght_data_start <= wght_data_start_pre;  // Commit start address
                wght_end_set    <= 0;                     // Clear flags for next activation
                wght_start_set  <= 0;

                if (wght_start_set) begin
                  wght_data_start <= wght_data_start_pre;
                  wght_data_vec <= wght_data_start_pre;
                end

                if (wght_end_set) begin
                  wght_data_end       <= wght_data_end_pre; // Commit end address
                  wght_data_start_pre <= wght_addr_SPad_data_r;
                end else begin
                  wght_data_end <= wght_addr_SPad_data_r;
                end

                // Determine next weight address vector based on activation sparsity
                if (iact_oh_delay_1 <= iact_oh_delay_2 + 1) begin
                  // Sequential weight addressing
                  if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                    wght_addr_vec <= wght_addr_vec + 1;
                  end
                  if (wght_end_set) begin
                    wght_data_start_pre <= wght_data_end_pre;
                  end else begin
                    wght_data_start_pre <= wght_addr_SPad_data_r;
                  end
                end else begin
                  // Sparse weight addressing (skip based on overhead bits)
                  if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                    wght_addr_vec <= iact_oh_delay_1 + 1;
                  end
                  wght_start_set <= 0;
                end

                if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                  wght_addr_vec <= wght_addr_vec + 1;
                end

                // Signal transition to next activation
                fast_cycle          <= 1;                     // Enable fast cycling
                iact_data_SPad_addr <= iact_data_SPad_addr + 1; // Move to next iact entry
                next_iact           <= 1;                     // Assert activation advance signal
                iact_addr_count     <= iact_addr_count + 1;   // Increment activation counter
              end

              // ================================================================
              // BLOCK 4: NEXT ACTIVATION HANDLING & DATA VALIDITY CHECK
              // ================================================================
              // Update activation data pipeline and validate current weight vector
              // Rationale: When advancing to next activation, update the pipeline
              // delays for proper data alignment. Also check if current weight
              // vector falls within the valid range for MAC operations.
              if (next_iact) begin
                // Update iact data pipeline (3-cycle delay for SPad read latency)
                iact_data_current_1  <= iact_data_spad_pay;  // Read new activation
                iact_data_current_2  <= iact_data_current_1; // Shift pipeline stage 1
                iact_data_current_3  <= iact_data_current_2; // Shift pipeline stage 2
                psum_spad_addr_a_mem <= 0;                   // Reset psum A memory addr
                psum_spad_addr_b_mem <= 1;                   // Reset psum B memory addr
                iact_addr_current    <= iact_addr_current + 1; // Increment activation index
              end

              // ================================================================
              // BLOCK 5: WEIGHT DATA VALIDITY CHECK
              // ================================================================
              // Determine if current weight vector should be included in MAC
              // Rationale: Data is valid only if current weight index is within
              // [start, end) range. Outside this range, no MAC operation occurs.
              values_valid <= 1;                              // Default: data is valid
              if (wght_data_end <= wght_data_vec) begin
                values_valid <= 0;                            // Weight vector out of range
              end
              // ================================================================
              // BLOCK 6: COMPUTATION COMPLETION CHECK
              // ================================================================
              // Detect when all activations have been processed
              // Rationale: When iact_addr_SPad_data_r == current+1, we've reached
              // the end of the activation sequence. Transition to output state.
              // Reuse Values of PSUM SPad
              if ((((iact_addr_SPad_data_r == iact_addr_current+1) | (iact_addr_count == 0)) & (wght_data_vec >= wght_data_end) | (iact_addr_count > iact_addr_SPad_data_r))) begin
                // All activations processed, prepare for psum output
                current_state_computing <= WAIT_TO_SEND_PSUM; // Transition state
                wght_addr_vec           <= 0;                  // Clear weight pointer
                wght_data_vec           <= 0;                  // Clear weight index
                wght_ready_o            <= 1;                  // Signal ready for new data
                mux_iact_ready          <= 1;                  // Signal ready status
                iact_data_current_3     <= 0;                  // Clear iact pipeline
                computing               <= 0;                  // Disable MAC operations
                psum_data_SPad_en_a_r   <= 0;                  // Disable psum reads
                psum_data_SPad_en_b_r   <= 0;
                psum_data_SPad_en_a_w   <= 1;                  // Enable psum writes
                psum_data_SPad_en_b_w   <= 1;
                values_valid            <= 0;                  // Invalidate current data
              end else begin
                // Continue computation, enable psum writes
                psum_data_SPad_en_a_w <= 1;                    // Write psum results
                psum_data_SPad_en_b_w <= 1;
              end

              // ================================================================
              // BLOCK 7: READ-AFTER-WRITE FORWARDING (Data Hazard Resolution)
              // ================================================================
              // Detect when a psum being read was just written, and forward the
              // newly computed value instead of reading from SPad
              // Rationale: Adders have 1-cycle latency. If we try to read a psum
              // that's being written in the same cycle, we need to forward the
              // result directly to avoid stalling or using stale data.
              if (psum_spad_addr_a_r == psum_spad_addr_a_w) begin
                // Port A read conflicts with Port A write
                reuse_psum_spad_a <= 1;       // Enable forwarding
                reused_data_a     <= adder_1_o_w; // Use adder 1 output
              end
              if (psum_spad_addr_b_r == psum_spad_addr_b_w) begin
                // Port B read conflicts with Port B write
                reuse_psum_spad_b <= 1;
                reused_data_b     <= adder_2_o_w;
              end
              if (!SERIAL) begin
                if (psum_spad_addr_a_r == psum_spad_addr_b_w) begin
                  reuse_psum_spad_a <= 1;
                  reused_data_a     <= adder_2_o_w;
                end
                if (psum_spad_addr_b_r == psum_spad_addr_a_w) begin
                  reuse_psum_spad_b <= 1;
                  reused_data_b     <= adder_1_o_w;
                end
              end
              
              // ================================================================
              // BLOCK 8: ADDER CONTROL AND PSUM MEMORY TRACKING
              // ================================================================
              // Enable MAC operations and track which psum locations have been
              // written to (to determine accumulate vs. first write)
              // Rationale: use_psum_x signals control whether the adder
              // accumulates with existing psum or starts fresh. Memory tracking
              // bits indicate if a psum location already contains a value.
              adder_1_en <= 1;                       // Enable multiplier/adder 1
              adder_2_en <= 1;                       // Enable multiplier/adder 2

              // Determine accumulation mode based on memory usage tracking
              if (SERIAL == 1) begin
                // Serial mode: separate tracking for each adder
                if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                  // Port A psum already written before: accumulate
                  use_psum_1 <= 1;
                end else begin
                  // Port A psum is new: first write
                  use_psum_1 <= 0;
                  used_psum_memory_1[(psum_spad_addr_a_r)] <= 1; // Mark as used
                end
                if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;
                end else begin
                  use_psum_2 <= 0;
                  used_psum_memory_2[(psum_spad_addr_b_r)] <= 1;
                end
              end else begin
                // Parallel mode: shared tracking for both adders
                if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1 <= 1;                   // Accumulate on port A
                end else begin
                  use_psum_1 <= 0;                   // First write on port A
                  used_psum_memory[(psum_spad_addr_a_r)] <= 1;
                end
                if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;                   // Accumulate on port B
                end else begin
                  use_psum_2 <= 0;                   // First write on port B
                  used_psum_memory[(psum_spad_addr_b_r)] <= 1;
                end
              end

              // ================================================================
              // BLOCK 8b: PSUM ADDRESS PIPELINE
              // ================================================================
              // Delay psum write addresses to match adder latency
              // Rationale: Adders have 1-cycle latency; we need to pipeline
              // addresses through registers to write to the correct location
              psum_spad_addr_a_delay <= psum_spad_addr_a_r; // Stage 1 delay
              psum_spad_addr_b_delay <= psum_spad_addr_b_r;
              psum_spad_addr_a_w     <= psum_spad_addr_a_delay; // Stage 2 delay
              psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
            //end
          end

          // ====================================================================
          // WAIT_TO_SEND_PSUM: Computation complete, wait for output enable
          // ====================================================================
          // Disables computation, prepares for partial sum output
          // Waits for psum_enable_i signal to begin streaming results
          WAIT_TO_SEND_PSUM: begin
            // Disable SPad reads for iact and wght (computation done)
            iact_addr_SPad_addr   <= 0;
            iact_addr_SPad_en_r   <= 0;

            iact_data_SPad_addr   <= 0;
            iact_data_SPad_en_r   <= 0;

            wght_addr_SPad_en_r   <= 0;
            wght_data_SPad_en_r   <= 0;

            psum_data_SPad_en_a_r <= 0;
            psum_data_SPad_en_b_r <= 0;
            psum_data_SPad_en_a_w <= 1;
            psum_data_SPad_en_b_w <= 1;
            psum_spad_addr_a_mem <= 0;
            if (SERIAL == 1) begin
              psum_spad_addr_b_mem <= 0;
            end else begin
              psum_spad_addr_b_mem <= 1;
            end
            psum_spad_addr_a_delay <= psum_spad_addr_a_r;
            psum_spad_addr_b_delay <= psum_spad_addr_b_r;
            psum_spad_addr_a_w     <= psum_spad_addr_a_delay;
            psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
            adder_1_en             <= computing;
            adder_2_en             <= computing;

            reuse_psum_spad_a      <= 0;
            reuse_psum_spad_b      <= 0;
            reused_data_a          <= 0;
            reused_data_b          <= 0;

            //Reuse Values of PSUM SPad
            if (psum_spad_addr_a_r == psum_spad_addr_a_w) begin
              reuse_psum_spad_a <= 1;
              reused_data_a     <= adder_1_o_w;
            end
            if (psum_spad_addr_b_r == psum_spad_addr_b_w) begin
              reuse_psum_spad_b <= 1;
              reused_data_b     <= adder_2_o_w;
            end
            if (psum_spad_addr_a_r == psum_spad_addr_b_w) begin
              reuse_psum_spad_a <= 1;
              reused_data_a     <= adder_2_o_w;
            end
            if (psum_spad_addr_b_r == psum_spad_addr_a_w) begin
              reuse_psum_spad_b <= 1;
              reused_data_b     <= adder_1_o_w;
            end

            if (psum_select) begin
              psum_data_SPad_en_a_w <= 0;
              psum_data_SPad_en_b_w <= 0;
              psum_spad_addr_a_mem  <= 0;
              if (SERIAL) begin
                psum_spad_addr_b_mem <= 0;
              end else begin
                psum_spad_addr_b_mem <= 1;
              end
              psum_spad_addr_a_w <= 2;
              psum_spad_addr_b_w <= 3;
            end
            if (psum_enable_i) begin
              adder_1_en              <= 1;
              adder_2_en              <= 1;
              current_state_computing <= SEND_PSUM;
              psum_data_SPad_en_a_w   <= 1;
              psum_data_SPad_en_b_w   <= 1;
              psum_spad_addr_a_w      <= psum_spad_addr_a_r;
              psum_spad_addr_b_w      <= psum_spad_addr_b_r;
              if (SERIAL) begin
                psum_spad_addr_a_mem <= psum_spad_addr_a_r + 1;
                psum_spad_addr_b_mem <= psum_spad_addr_b_r + 1;
                if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1                               <= 1;
                  used_psum_memory_1[(psum_spad_addr_a_r)] <= 0;
                end else begin
                  use_psum_1 <= 0;
                end
                if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2                               <= 1;
                  used_psum_memory_2[(psum_spad_addr_b_r)] <= 0;
                end else begin
                  use_psum_2 <= 0;
                end
              end else begin
                psum_spad_addr_a_mem <= psum_spad_addr_a_r + 2;
                psum_spad_addr_b_mem <= psum_spad_addr_b_r + 2;
                if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1                             <= 1;
                  used_psum_memory[(psum_spad_addr_a_r)] <= 0;
                end else begin
                  use_psum_1 <= 0;
                end
                if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2                             <= 1;
                  used_psum_memory[(psum_spad_addr_b_r)] <= 0;
                end else begin
                  use_psum_2 <= 0;
                end
              end
            end else begin
              if (SERIAL == 1) begin
                if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1 <= 1;
                end else begin
                  use_psum_1 <= 0;
                  //used_psum_memory_1[(psum_spad_addr_a_r)] <= 1;
                end
                if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;
                end else begin
                  use_psum_2 <= 0;
                  //used_psum_memory_2[(psum_spad_addr_b_r)] <= 1;
                end
              end else begin
                if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1 <= 1;
                end else begin
                  use_psum_1 <= 0;
                  used_psum_memory[(psum_spad_addr_a_r)] <= 1;
                end
                if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;
                end else begin
                  use_psum_2 <= 0;
                  used_psum_memory[(psum_spad_addr_b_r)] <= 1;
                end
              end
            end
            computing   <= 0;
            psum_select <= !computing;
          end

          // ====================================================================
          // SEND_PSUM: Stream partial sum results to output
          // ====================================================================
          // Reads psum values from SPad memory and outputs them
          // Clears usage bitmap as values are sent
          // Returns to IDLE when psum_enable_i deasserts
          SEND_PSUM: begin
            psum_spad_addr_a_w <= psum_spad_addr_a_r;
            psum_spad_addr_b_w <= psum_spad_addr_b_r;
            adder_1_en         <= 1;
            adder_2_en         <= 1;
            psum_select        <= !computing;
            // When psum output completes, return to IDLE
            if (!psum_enable_i) begin
              current_state_computing <= IDLE;
              adder_1_en              <= 1;
              adder_2_en              <= 1;
            end
            if (SERIAL == 1) begin
              psum_spad_addr_a_mem <= psum_spad_addr_a_r + 1;
              psum_spad_addr_b_mem <= psum_spad_addr_b_r + 1;
              adder_3_en           <= 1;
              if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1                               <= 1;
                used_psum_memory_1[(psum_spad_addr_a_r)] <= 0;
              end else begin
                use_psum_1 <= 0;
              end
              if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2                               <= 1;
                used_psum_memory_2[(psum_spad_addr_b_r)] <= 0;
              end else begin
                use_psum_2 <= 0;
              end
            end else begin
              psum_spad_addr_a_mem <= psum_spad_addr_a_r + 2;
              psum_spad_addr_b_mem <= psum_spad_addr_b_r + 2;
              if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1                             <= 1;
                used_psum_memory[(psum_spad_addr_a_r)] <= 0;
              end else begin
                use_psum_1 <= 0;
              end
              if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2                             <= 1;
                used_psum_memory[(psum_spad_addr_b_r)] <= 0;
              end else begin
                use_psum_2 <= 0;
              end
            end
          end
          default: begin
          end
        endcase
      end
    end
  end else begin : gen_dense_fsm
    // ============================================================================
    // DENSE FSM (SPARSITY_EN=0)
    // ============================================================================
    // Simplified FSM for dense (non-sparse) computation:
    // - No weight address SPad needed (weights stored densely)
    // - No overhead-bit pipeline (iact_oh_delay_*, iact_data_spad_oh unused)
    // - No weight range computation (all weights valid)
    // - No zero-skipping (next_iact/values_valid unused)
    // - Sequential addressing: iact and weight SPads are read in order
    // - Loop termination: when all iact data words processed

    reg  [          IACT_ADDR_DATA-1 : 0] iact_channel;     // Current iact address being processed
    reg  [          IACT_ADDR_DATA-1 : 0] wght_filter;     // Current iact address being processed
    reg                                   computing_1;              // Flag indicating active computation
    reg                                   computing_2;              // Flag indicating active computation
    always @(posedge clk_i, negedge rst_ni) begin
      if (!rst_ni) begin
        current_state_computing <= IDLE;
        iact_addr_SPad_addr     <= 0;
        iact_addr_SPad_en_r     <= 0;
        iact_data_SPad_addr     <= 0;
        iact_data_SPad_en_r     <= 0;
        wght_addr_SPad_en_r     <= 0;
        wght_data_vec           <= 0;
        wght_data_SPad_en_r     <= 0;
        psum_data_SPad_en_a_r   <= 0;
        psum_data_SPad_en_b_r   <= 0;
        psum_data_SPad_en_a_w   <= 0;
        psum_data_SPad_en_b_w   <= 0;
        iact_data_current_3     <= 0;
        computing               <= 0;
        computing_1             <= 0;
        computing_2             <= 0;
        fast_cycle              <= 0;
        used_psum_memory        <= 0;
        use_psum_1              <= 0;
        use_psum_2              <= 0;
        adder_1_en              <= 0;
        adder_2_en              <= 0;
        iact_channel            <= 0;
        wght_filter             <= 0;
        mux_iact_ready          <= 1;
        wght_ready_o            <= 1;
        psum_select             <= 1;
        psum_data_1_delay       <= 0;
        psum_data_2_delay       <= 0;
        psum_enable             <= 0;
        psum_enable_2           <= 0;
        psum_enable_o           <= 0;
        psum_spad_addr_a_mem    <= 0;
        psum_spad_addr_b_mem    <= 1;
        reuse_psum_spad_a       <= 0;
        reuse_psum_spad_b       <= 0;
        reused_data_a           <= 0;
        reused_data_b           <= 0;
        psum_spad_addr_a_delay  <= 0;
        psum_spad_addr_b_delay  <= 1;
        psum_spad_addr_a_w      <= 0;
        psum_spad_addr_b_w      <= 1;
        // Serial-mode psum memory
        used_psum_memory_1      <= 0;
        used_psum_memory_2      <= 0;
        values_valid            <= 0;
      end else begin
        // Psum pipeline and external psum input handling (identical to sparse FSM)
        if (psum_ready_i) begin
          psum_enable <= psum_enable_i;
        end else begin
          psum_enable <= 0;
        end
        {psum_data_2_delay, psum_data_1_delay} <= {{(TRANS_BITWIDTH_PSUM) {1'd0}}, psum_data_i};
        if (SERIAL == 1) begin
          psum_enable_2 <= psum_enable;
          psum_enable_o <= psum_enable_2;
        end else begin
          psum_enable_o <= psum_enable;
        end
        case (current_state_computing)
          // ==================================================================
          // IDLE: Wait for compute trigger or psum read request
          // ==================================================================
          IDLE: begin
            mux_iact_ready         <= 1;
            wght_ready_o           <= 1;
            fast_cycle             <= 0;
            iact_addr_SPad_addr    <= 0;
            iact_addr_SPad_en_r    <= 0;
            iact_data_SPad_addr    <= 0;
            iact_data_SPad_en_r    <= 0;
            wght_addr_SPad_en_r    <= 0;
            wght_data_vec          <= 0;
            wght_data_SPad_en_r    <= 0;
            psum_data_SPad_en_a_r  <= computing;
            psum_data_SPad_en_b_r  <= computing;
            psum_data_SPad_en_a_w  <= 0;
            psum_data_SPad_en_b_w  <= 0;
            computing              <= 0;
            computing_1            <= 0;
            computing_2            <= 0;
            psum_spad_addr_a_delay <= 0;
            psum_spad_addr_b_delay <= 1;
            psum_spad_addr_a_w     <= 0;
            psum_spad_addr_b_w     <= 1;
            psum_spad_addr_a_mem   <= 0;
            psum_spad_addr_b_mem   <= 1;
            adder_1_en             <= 0;
            adder_2_en             <= 0;
            adder_3_en             <= 0;
            reuse_psum_spad_a      <= 0;
            reuse_psum_spad_b      <= 0;
            reused_data_a          <= 0;
            reused_data_b          <= 0;
            use_psum_1             <= 0;
            use_psum_2             <= 0;
            psum_select            <= 1;
            values_valid           <= 0;
            if (SERIAL == 1) begin
              used_psum_memory_1 <= 0;
              used_psum_memory_2 <= 0;
            end else begin
              used_psum_memory <= 0;
            end
            // Psum read-out request
            if (psum_enable_i) begin
              current_state_computing <= SEND_PSUM;
              adder_1_en              <= 1;
              adder_2_en              <= 1;
              psum_select             <= 1;
              use_psum_1              <= 0;
              use_psum_2              <= 0;
              if (SERIAL == 1) begin
                used_psum_memory_1 <= 0;
                used_psum_memory_2 <= 0;
              end else begin
                used_psum_memory <= 0;
              end
            end
            // Start computation when both iact and wght data are loaded
            if (data_set & compute_i & ((second_spad_words_iact != 0) & (second_spad_words_wght != 0))) begin
              current_state_computing <= LOADING_1;
              mux_iact_ready          <= 0;
              wght_ready_o            <= 0;
              psum_select             <= 0;
              iact_data_SPad_addr     <= 0;
              iact_data_SPad_en_r     <= 1;
              wght_data_vec           <= 0;
              wght_data_SPad_en_r     <= 1;
              psum_data_SPad_en_a_r   <= 0;
              psum_data_SPad_en_b_r   <= 0;
              use_psum_1              <= 0;
              use_psum_2              <= 0;
              if (SERIAL == 1) begin
                used_psum_memory_1 <= 0;
                used_psum_memory_2 <= 0;
              end else begin
                used_psum_memory <= 0;
              end
            end
          end

          // ==================================================================
          // LOADING_1: Pipeline fill — 1 cycles to fill iact/wght data path
          // ==================================================================
          // In dense mode, we don't need to fetch weight addresses from a
          // separate SPad. We simply start reading iact and wght data SPads
          // sequentially and fill the 3-stage iact pipeline.
          LOADING_1: begin
            current_state_computing <= CALCULATING;
            iact_data_current_3     <= iact_data_spad_pay;
            if (0 >=  filters_reg_M0 - PARALLEL_MACS) begin
              iact_data_SPad_addr     <= iact_data_SPad_addr + 1;
            end else begin
              wght_filter <= wght_filter + PARALLEL_MACS;
            end
          end

          // ==================================================================
          // CALCULATING: Main dense MAC loop
          // ==================================================================
          // All weights are valid (no zero-skipping). Sequential addressing:
          // - iact: one new activation per PARALLEL_MACS weight pairs
          // - wght: increment by 1 each cycle (both MACs fed from wght_data_vec)
          // Completion: when wght_data_vec wraps through all weight entries
          CALCULATING: begin
            // ---------------------------------------------------------------
            // Default signal setup
            // ---------------------------------------------------------------
            computing             <= 1;
            computing_1           <= 1;
            computing_2           <= 1;
            iact_data_current_3   <= iact_data_spad_pay;
            iact_addr_SPad_en_r   <= 0;
            iact_data_SPad_en_r   <= !mux_iact_ready;
            wght_addr_SPad_en_r   <= 0;           // No weight addr SPad in dense mode
            psum_data_SPad_en_a_r <= computing;
            psum_data_SPad_en_b_r <= computing;
            psum_data_SPad_en_a_w <= psum_data_SPad_en_a_r;
            psum_data_SPad_en_b_w <= psum_data_SPad_en_b_r;
            reuse_psum_spad_a     <= 0;
            reuse_psum_spad_b     <= 0;
            reused_data_a         <= 0;
            reused_data_b         <= 0;
            fast_cycle            <= 0;
            values_valid          <= 1; // Default: data is valid

            // ---------------------------------------------------------------
            // Weight vector increment (sequential, no range computation)
            // ---------------------------------------------------------------
            if (wght_data_vec < (second_spad_words_wght - 1)) begin
              wght_data_vec <= wght_data_vec + 1;
            end else begin
              wght_data_vec  <= 0;
              mux_iact_ready <= 1;
            end

            // ---------------------------------------------------------------
            // Iact pipeline advance (every PARALLEL_MACS weight steps)
            // One new iact value feeds both multipliers simultaneously.
            // In dense mode the iact pipeline shifts every cycle because
            // each iact pairs with PARALLEL_MACS weight values. When all
            // weights for the current iact have been consumed, advance iact.
            // ---------------------------------------------------------------
            if (mux_iact_ready) begin
              iact_data_current_3  <= 0;
              iact_data_SPad_addr  <= iact_data_SPad_addr + 1;
              mux_iact_ready       <= 0;
              psum_spad_addr_a_mem <= 0;
              psum_spad_addr_b_mem <= 1;
            end

            // ---------------------------------------------------------------
            // Completion: all iact data words processed
            // ---------------------------------------------------------------

            wght_filter <= wght_filter + PARALLEL_MACS;
            if (wght_filter >= filters_reg_M0 - PARALLEL_MACS) begin
              wght_filter          <= 0;
              iact_channel         <= iact_channel + 1;
              iact_data_SPad_addr  <= iact_data_SPad_addr + 1;
              if (((channel_reg_C0 * first_spad_words_iact_S) - 1 == iact_channel )) begin
                iact_channel            <= 0;
                current_state_computing <= WAIT_TO_SEND_PSUM;
                wght_ready_o            <= 1;
                mux_iact_ready          <= 1;
                psum_data_SPad_en_a_w   <= 1;
                psum_data_SPad_en_b_w   <= 1;
              end else begin
                psum_data_SPad_en_a_w <= 1;
                psum_data_SPad_en_b_w <= 1;
              end
            end
            psum_spad_addr_a_mem  <= psum_spad_addr_b_r + 1;
            psum_spad_addr_b_mem  <= psum_spad_addr_b_r + 2;
            if (wght_filter == 2) begin
              psum_spad_addr_a_mem <= 0;
              psum_spad_addr_b_mem <= 1;
            end

            // ---------------------------------------------------------------
            // Adder enable and psum memory tracking (identical to sparse FSM)
            // ---------------------------------------------------------------
            adder_1_en <= 1;
            adder_2_en <= 1;
            if (SERIAL == 1) begin
              if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1 <= 1;
              end else begin
                use_psum_1 <= 0;
                used_psum_memory_1[(psum_spad_addr_a_r)] <= 1;
              end
              if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2 <= 1;
              end else begin
                use_psum_2 <= 0;
                used_psum_memory_2[(psum_spad_addr_b_r)] <= 1;
              end
            end else begin
              if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1 <= 1;
              end else begin
                use_psum_1 <= 0;
                used_psum_memory[(psum_spad_addr_a_r)] <= 1;
              end
              if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2 <= 1;
              end else begin
                use_psum_2 <= 0;
                used_psum_memory[(psum_spad_addr_b_r)] <= 1;
              end
            end

            // ---------------------------------------------------------------
            // Psum address pipeline (identical to sparse FSM)
            // ---------------------------------------------------------------
            psum_spad_addr_a_delay <= psum_spad_addr_a_r;
            psum_spad_addr_b_delay <= psum_spad_addr_b_r;
            psum_spad_addr_a_w     <= psum_spad_addr_a_delay;
            psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
          end

          // ==================================================================
          // WAIT_TO_SEND_PSUM: Computation done, wait for output enable
          // ==================================================================
          WAIT_TO_SEND_PSUM: begin
            computing             <= 0;
            computing_1           <= computing;
            computing_2           <= computing_1;
            iact_addr_SPad_addr   <= computing;
            iact_addr_SPad_en_r   <= 0;
            iact_data_SPad_addr   <= 0;
            iact_data_SPad_en_r   <= 0;
            wght_addr_SPad_en_r   <= 0;
            wght_data_SPad_en_r   <= 0;
            psum_data_SPad_en_a_r <= computing;
            psum_data_SPad_en_b_r <= computing;
            psum_data_SPad_en_a_w <= 1;
            if (computing) begin
              psum_spad_addr_a_mem <= psum_spad_addr_a_mem + 2;
              psum_spad_addr_b_mem <= psum_spad_addr_b_mem + 2;
            end else begin
              psum_spad_addr_a_mem <= 0;
              if (SERIAL == 1) begin
                psum_spad_addr_b_mem <= 0;
              end else begin
                psum_spad_addr_b_mem <= 1;
              end
            end
            psum_spad_addr_a_delay <= psum_spad_addr_a_r;
            psum_spad_addr_b_delay <= psum_spad_addr_b_r;
            psum_spad_addr_a_w     <= 2;
            psum_spad_addr_b_w     <= 3;
            if (computing_2) begin
              psum_spad_addr_a_w     <= psum_spad_addr_a_delay;
              psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
            end
            adder_1_en             <= values_valid;
            adder_2_en             <= values_valid;
            reuse_psum_spad_a      <= 0;
            reuse_psum_spad_b      <= 0;
            reused_data_a          <= 0;
            reused_data_b          <= 0;
            if (psum_spad_addr_a_r == psum_spad_addr_a_w) begin
              reuse_psum_spad_a <= 1;
              reused_data_a     <= adder_1_o_w;
            end
            if (psum_spad_addr_b_r == psum_spad_addr_b_w) begin
              reuse_psum_spad_b <= 1;
              reused_data_b     <= adder_2_o_w;
            end
            if (psum_spad_addr_a_r == psum_spad_addr_b_w) begin
              reuse_psum_spad_a <= 1;
              reused_data_a     <= adder_2_o_w;
            end
            if (psum_spad_addr_b_r == psum_spad_addr_a_w) begin
              reuse_psum_spad_b <= 1;
              reused_data_b     <= adder_1_o_w;
            end
            if (psum_select) begin
              psum_data_SPad_en_a_w <= 0;
              psum_data_SPad_en_b_w <= 0;
              psum_spad_addr_a_mem  <= 0;
              if (SERIAL) begin
                psum_spad_addr_b_mem <= 0;
              end else begin
                psum_spad_addr_b_mem <= 1;
              end
              iact_data_current_3 <= 0;
              values_valid        <= 0;
            end
            if (psum_enable_i) begin
              adder_1_en              <= 1;
              adder_2_en              <= 1;
              current_state_computing <= SEND_PSUM;
              psum_data_SPad_en_a_w   <= 1;
              psum_data_SPad_en_b_w   <= 1;
              psum_spad_addr_a_w      <= psum_spad_addr_a_r;
              psum_spad_addr_b_w      <= psum_spad_addr_b_r;
              if (SERIAL) begin
                psum_spad_addr_a_mem <= psum_spad_addr_a_r + 1;
                psum_spad_addr_b_mem <= psum_spad_addr_b_r + 1;
                if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1                               <= 1;
                  used_psum_memory_1[(psum_spad_addr_a_r)] <= 0;
                end else begin
                  use_psum_1 <= 0;
                end
                if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2                               <= 1;
                  used_psum_memory_2[(psum_spad_addr_b_r)] <= 0;
                end else begin
                  use_psum_2 <= 0;
                end
              end else begin
                psum_spad_addr_a_mem <= psum_spad_addr_a_r + 2;
                psum_spad_addr_b_mem <= psum_spad_addr_b_r + 2;
                if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1                             <= 1;
                  used_psum_memory[(psum_spad_addr_a_r)] <= 0;
                end else begin
                  use_psum_1 <= 0;
                end
                if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2                             <= 1;
                  used_psum_memory[(psum_spad_addr_b_r)] <= 0;
                end else begin
                  use_psum_2 <= 0;
                end
              end
            end else begin
              if (SERIAL == 1) begin
                if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1 <= 1;
                end else begin
                  use_psum_1 <= 0;
                end
                if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;
                end else begin
                  use_psum_2 <= 0;
                end
              end else begin
                if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                  use_psum_1 <= 1;
                end else begin
                  use_psum_1 <= 0;
                  used_psum_memory[(psum_spad_addr_a_r)] <= 1;
                end
                if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                  use_psum_2 <= 1;
                end else begin
                  use_psum_2 <= 0;
                  used_psum_memory[(psum_spad_addr_b_r)] <= 1;
                end
              end
            end
            psum_select <= !computing_1;
          end

          // ==================================================================
          // SEND_PSUM: Stream results to output (identical to sparse FSM)
          // ==================================================================
          SEND_PSUM: begin
            psum_spad_addr_a_w <= psum_spad_addr_a_r;
            psum_spad_addr_b_w <= psum_spad_addr_b_r;
            adder_1_en         <= 1;
            adder_2_en         <= 1;
            psum_select        <= !computing;
            if (!psum_enable_i) begin
              current_state_computing <= IDLE;
              adder_1_en              <= 1;
              adder_2_en              <= 1;
            end
            if (SERIAL == 1) begin
              psum_spad_addr_a_mem <= psum_spad_addr_a_r + 1;
              psum_spad_addr_b_mem <= psum_spad_addr_b_r + 1;
              adder_3_en           <= 1;
              if (used_psum_memory_1[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1                               <= 1;
                used_psum_memory_1[(psum_spad_addr_a_r)] <= 0;
              end else begin
                use_psum_1 <= 0;
              end
              if (used_psum_memory_2[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2                               <= 1;
                used_psum_memory_2[(psum_spad_addr_b_r)] <= 0;
              end else begin
                use_psum_2 <= 0;
              end
            end else begin
              psum_spad_addr_a_mem <= psum_spad_addr_a_r + 2;
              psum_spad_addr_b_mem <= psum_spad_addr_b_r + 2;
              if (used_psum_memory[(psum_spad_addr_a_r)] == 1) begin
                use_psum_1                             <= 1;
                used_psum_memory[(psum_spad_addr_a_r)] <= 0;
              end else begin
                use_psum_1 <= 0;
              end
              if (used_psum_memory[(psum_spad_addr_b_r)] == 1) begin
                use_psum_2                             <= 1;
                used_psum_memory[(psum_spad_addr_b_r)] <= 0;
              end else begin
                use_psum_2 <= 0;
              end
            end
          end
          default: begin
          end
        endcase
      end
    end
  end endgenerate // gen_sparse_fsm / gen_dense_fsm

  // ============================================================================
  // Scratch Pad Memory Instantiations
  // ============================================================================

  // Input Activation Address SPad (single-port)
  // Stores addresses/indices for sparse input activation data
  // Only instantiated when SPARSITY_EN=1
  generate
    if (SPARSITY_EN == 1) begin : gen_iact_addr_spad
      SPad_SP #(
          .DATA_WIDTH(IACT_ADDR_DATA),
          .ADDR_WIDTH(IACT_ADDR_ADDR_BITWIDTH)
      ) iact_addr_SPad (
          .clk_i (clk_i),
          .re_i  (iact_addr_SPad_en_r & !first_spad_iact_en_w),
          .we_i  (first_spad_iact_en_w),
          .addr_i(iact_addr_SPad_addr | first_spad_iact_addr_w),
          .data_i(first_spad_iact_data_w),
          .data_o(iact_addr_SPad_data_r)
      );
    end else begin : gen_iact_addr_spad_dummy
      // Dense mode: no address SPAD, tie output to 0
      assign iact_addr_SPad_data_r = {IACT_ADDR_DATA{1'b0}};
    end
  endgenerate

  // Input Activation Data SPad (single-port)
  // Stores actual input activation values with overhead bits for sparsity
  SPad_SP #(
      .DATA_WIDTH(IACT_DATA_DATA),
      .ADDR_WIDTH(IACT_DATA_ADDR_BITWIDTH)
  ) iact_data_SPad (
      .clk_i (clk_i),
      .re_i  (iact_data_SPad_en_r & !second_spad_iact_en_w),
      .we_i  (second_spad_iact_en_w),
      .addr_i(iact_data_SPad_addr | second_spad_iact_addr_w),
      .data_i(second_spad_iact_data_w),
      .data_o(iact_data_SPad_data_r)
  );

  // Weight Address SPad (single-port)
  // Stores pointers/addresses into weight data SPad for sparse weight access
  // Only instantiated when SPARSITY_EN=1
  generate
    if (SPARSITY_EN == 1) begin : gen_wght_addr_spad
      SPad_SP #(
          .DATA_WIDTH(WGHT_ADDR_DATA),
          .ADDR_WIDTH(WGHT_ADDR_ADDR_BITWIDTH)
      ) weight_addr_SPad (
          .clk_i (clk_i),
          .re_i  (wght_addr_SPad_en_r & !first_spad_wght_en_w),
          .we_i  (first_spad_wght_en_w),
          .addr_i(wght_addr_SPad_addr | first_spad_wght_addr_w),
          .data_i(first_spad_wght_data_w),
          .data_o(wght_addr_SPad_data_r)
      );
    end else begin : gen_wght_addr_spad_dummy
      // Dense mode: no address SPAD, tie output to 0
      assign wght_addr_SPad_data_r = {WGHT_ADDR_DATA{1'b0}};
    end
  endgenerate

  // Weight Data SPad (single-port)
  // Stores actual weight values (parallel sets for dual MACs) with overhead bits
  SPad_SP #(
      .DATA_WIDTH(WGHT_DATA_DATA),
      .ADDR_WIDTH(WGHT_DATA_ADDR_BITWIDTH)
  ) weight_data_SPad (
      .clk_i (clk_i),
      .re_i  (wght_data_SPad_en_r & !second_spad_wght_en_w),
      .we_i  (second_spad_wght_en_w),
      .addr_i(wght_data_SPad_addr | second_spad_wght_addr_w),
      .data_i(second_spad_wght_data_w),
      .data_o(wght_data_SPad_data_r)
  );

  // Partial Sum SPads (dual-port for read-write concurrency)
  // Serial mode: Two separate dual-port SPads (one per MAC)
  // Parallel mode: One dual-port SPad with dual read/write capability
  if (SERIAL) begin : gen_serial_spad
    SPad_DP #(
        .DATA_WIDTH(PSUM_DATA),
        .ADDR_WIDTH(PSUM_ADDR_BITWIDTH)
    ) psum_SPad_A (
        .clk_i(clk_i),
        .re_i(psum_data_SPad_en_a_r || psum_enable_i),
        .we_i(psum_data_SPad_en_a_w_i),
        .addr_r_i(psum_spad_addr_a_r),
        .addr_w_i(psum_spad_addr_a_w),
        .data_i(psum_spad_data_a_i),
        .data_o(psum_spad_data_a_o)
    );
    SPad_DP #(
        .DATA_WIDTH(PSUM_DATA),
        .ADDR_WIDTH(PSUM_ADDR_BITWIDTH)
    ) psum_SPad_B (
        .clk_i(clk_i),
        .re_i(psum_data_SPad_en_b_r || psum_enable_i),
        .we_i(psum_data_SPad_en_b_w_i),
        .addr_r_i(psum_spad_addr_b_r),
        .addr_w_i(psum_spad_addr_b_w),
        .data_i(psum_spad_data_b_i),
        .data_o(psum_spad_data_b_o)
    );
  end else begin : gen_parallel_spad
    SPad_DP_RW #(
        .DATA_WIDTH(PSUM_DATA),
        .ADDR_WIDTH(PSUM_ADDR_BITWIDTH)
    ) psum_SPad (
        .clk_i     (clk_i),
        .re_a_i    (psum_data_SPad_en_a_r || psum_enable_i),
        .re_b_i    (psum_data_SPad_en_b_r || psum_enable_i),
        .we_a_i    (psum_data_SPad_en_a_w_i),
        .we_b_i    (psum_data_SPad_en_b_w_i),
        .addr_r_a_i(psum_spad_addr_a_r),
        .addr_r_b_i(psum_spad_addr_b_r),
        .addr_w_a_i(psum_spad_addr_a_w),
        .addr_w_b_i(psum_spad_addr_b_w),
        .data_a_i  (psum_spad_data_a_i),
        .data_b_i  (psum_spad_data_b_i),
        .data_a_o  (psum_spad_data_a_o),
        .data_b_o  (psum_spad_data_b_o)
    );
  end

  // ============================================================================
  // Data Path Modules
  // ============================================================================

  // Input Activation Multiplexer
  // Selects which of the NUM_GLB_IACT global buffer inputs to use
  mux_iact #(
      .WIDTH  (TRANS_BITWIDTH_IACT),
      .I_COUNT(NUM_GLB_IACT)
  ) mux_iact (
      .a_i  (iact_data_i),
      .b_i  (iact_enable_i),
      .c_i  (mux_iact_c_i_w),
      .sel_i(iact_select_i),
      .a_o  (mux_iact_a_o_w),
      .b_o  (mux_iact_b_o_w),
      .c_o  (iact_ready_o)
  );

  // Weight Data Pipeline
  // Receives weight data from external interface and unpacks it into
  // two-level SPad structure (address SPad + data SPad) for sparse storage
  data_pipeline_wght #(
      .DATA_WIDTH      (TRANS_BITWIDTH_WGHT),
      .SPARSITY_EN     (SPARSITY_EN),
      .FIRST_SPAD_ADDR (WGHT_ADDR_ADDR),
      .FIRST_SPAD_DATA (WGHT_ADDR_DATA),
      .SECOND_SPAD_ADDR(WGHT_DATA_ADDR),
      .SECOND_SPAD_DATA(WGHT_DATA_DATA)
  ) wght_data_handler (
      .clk_i    (clk_i),
      .rst_ni   (rst_ni),
      .compute_i(compute_i | enable_stream_i),

      .data_i  (wght_data_i),
      .enable_i(wght_enable_i),

      .first_spad_words_o (first_spad_words_wght),
      .first_spad_max_i   (filters_reg_M0),
      .second_spad_words_o(second_spad_words_wght),

      .first_spad_addr_o(first_spad_wght_addr_w),
      .first_spad_data_o(first_spad_wght_data_w),
      .first_spad_en_o  (first_spad_wght_en_w),

      .second_spad_addr_o(second_spad_wght_addr_w),
      .second_spad_data_o(second_spad_wght_data_w),
      .second_spad_en_o  (second_spad_wght_en_w),

      // Raw (dense/GEMM) weight streams carry legitimate all-zero words
      .raw_mode_i(raw_wght_reg)
  );

  // Input Activation Data Pipeline
  // Receives iact data from multiplexer and unpacks it into
  // two-level SPad structure (address SPad + data SPad) for sparse storage
  data_pipeline_iact #(
      .DATA_WIDTH      (TRANS_BITWIDTH_IACT),
      //.SPARSITY_EN     (SPARSITY_EN),
      .FIRST_SPAD_ADDR (IACT_ADDR_ADDR),
      .FIRST_SPAD_DATA (IACT_ADDR_DATA),
      .SECOND_SPAD_ADDR(IACT_DATA_ADDR),
      .SECOND_SPAD_DATA(IACT_DATA_DATA)
  ) iact_data_handler (
      .clk_i    (clk_i),
      .rst_ni   (rst_ni),
      .compute_i(compute_i | enable_stream_i),

      .data_i                    (mux_iact_a_o_w),
      .enable_i                  (mux_iact_b_o_w),
      .iact_x_line_repetitions_i (iact_x_line_repetitions),

      .first_spad_words_o        (first_spad_words_iact_S),
      .first_spad_max_i          (channel_reg_C0),
      .second_spad_words_o       (second_spad_words_iact),

      .first_spad_addr_o         (first_spad_iact_addr_w),
      .first_spad_data_o         (first_spad_iact_data_w),
      .first_spad_en_o           (first_spad_iact_en_w),

      .second_spad_addr_o        (second_spad_iact_addr_w),
      .second_spad_data_o        (second_spad_iact_data_w),
      .second_spad_en_o          (second_spad_iact_en_w)
  );

  // ============================================================================
  // Computational Units (Multipliers and Adders)
  // ============================================================================
  // USE_DSP parameter controls MAC implementation:
  //   USE_DSP = 0: Standard multiplier + adder pipeline (fabric-based, flexible)
  //   USE_DSP = 1: DSP48 slice optimization (integrated MAC, FPGA-optimized)

  generate
    if (USE_DSP == 0) begin : gen_standard_mac
      // ========================================================================
      // Standard Implementation: Separate Multipliers and Adders
      // ========================================================================

      // Multiplier 1: First parallel MAC unit
      // Multiplies weight_1 * input_activation (fixed-point arithmetic)
      multiplier #(
          .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
          .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
          .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH)
      ) multiplier_1 (
          .clk_i          (clk_i),
          .rst_ni         (rst_ni),
          .multiplier_en_i(values_valid),
          .factor_1       (mult_1_fac_1),
          .factor_2       (mult_1_fac_2),
          .product        (mult_1_o_w)
      );

      // Multiplier 2: Second parallel MAC unit
      // Multiplies weight_2 * input_activation (same iact as multiplier 1)
      multiplier #(
          .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
          .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
          .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH)
      ) multiplier_2 (
          .clk_i          (clk_i),
          .rst_ni         (rst_ni),
          .multiplier_en_i(values_valid),
          .factor_1       (mult_2_fac_1),
          .factor_2       (mult_2_fac_2),
          .product        (mult_2_o_w)
      );

      // Adder 1: Accumulator for MAC 1
      // Adds multiplier 1 output to partial sum (for accumulation)
      adder #(
          .DATA_WIDTH_SUM(DATA_PSUM_BITWIDTH)
      ) adder_1 (
          .clk_i      (clk_i),
          .rst_ni     (rst_ni),
          .summand_1_i(adder_1_summand_1),
          .summand_2_i(adder_1_summand_2),
          .sum_o      (adder_1_o_w),
          .adder_en_i (adder_1_en)
      );

      // Adder 2: Accumulator for MAC 2
      // Adds multiplier 2 output to partial sum (for accumulation)
      adder #(
          .DATA_WIDTH_SUM(DATA_PSUM_BITWIDTH)
      ) adder_2 (
          .clk_i      (clk_i),
          .rst_ni     (rst_ni),
          .summand_1_i(adder_2_summand_1),
          .summand_2_i(adder_2_summand_2),
          .sum_o      (adder_2_o_w),
          .adder_en_i (adder_2_en)
      );

    end else begin : gen_dsp_mac
      // ========================================================================
      // DSP48 Implementation: Integrated Multiply-Accumulate Units
      // ========================================================================

      // DSP Unit 1: First parallel MAC unit with integrated DSP48 slice
      // Performs multiply-accumulate: (weight_1 * iact) + psum in single DSP slice
      dsp_unit #(
          .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
          .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
          .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH),
          .DATA_WIDTH_ADDI(DATA_PSUM_BITWIDTH)
      ) dsp_unit_1 (
          .clk_i      (clk_i),
          .rst_ni     (rst_ni),
          .multi_en_i (values_valid),
          .adder_en_i (adder_1_en),
          .adder_sel_i(psum_select),
          .a_in       (mult_1_fac_1),
          .b_in       (mult_1_fac_2),
          .c_in       (adder_1_summand_1),
          .d_in       (psum_data_1_delay[DATA_PSUM_BITWIDTH-1:0]),
          .p_out      (adder_1_o_w)
      );

      // DSP Unit 2: Second parallel MAC unit with integrated DSP48 slice
      // Performs multiply-accumulate: (weight_2 * iact) + psum in single DSP slice
      dsp_unit #(
          .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
          .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
          .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH),
          .DATA_WIDTH_ADDI(DATA_PSUM_BITWIDTH)
      ) dsp_unit_2 (
          .clk_i      (clk_i),
          .rst_ni     (rst_ni),
          .multi_en_i (values_valid),
          .adder_en_i (adder_2_en),
          .adder_sel_i(psum_select),
          .a_in       (mult_2_fac_1),
          .b_in       (mult_2_fac_2),
          .c_in       (adder_2_summand_1),
          .d_in       (psum_data_2_delay[DATA_PSUM_BITWIDTH-1:0]),
          .p_out      (adder_2_o_w)
      );

      // Note: In DSP mode, mult_1_o_w and mult_2_o_w are not used
      // The DSP units output directly to adder_1_o_w and adder_2_o_w
      assign mult_1_o_w = {DATA_PSUM_BITWIDTH{1'b0}};
      assign mult_2_o_w = {DATA_PSUM_BITWIDTH{1'b0}};
    end
  endgenerate

  // Serial mode only: Adder 3 combines outputs from adders 1 and 2
  // This is common to both USE_DSP=0 and USE_DSP=1 modes
  if (SERIAL) begin : gen_serial_adder
    // Adder 3: Combines partial sums from both MACs (serial mode only)
    adder #(
        .DATA_WIDTH_SUM(DATA_PSUM_BITWIDTH)
    ) adder_3 (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .summand_1_i(adder_3_summand_1),
        .summand_2_i(adder_3_summand_2),
        .sum_o      (adder_3_o_w),
        .adder_en_i (adder_3_en)
    );
  end

  // ============================================================================
  // Partial Sum Input Multiplexer
  // ============================================================================
  // Selects between psum from SPad (for accumulation) or external psum (from router/other PE)
  if (SERIAL) begin : gen_serial_psum_multiplexer
    mux2 #(
        .DATA_WIDTH(TRANS_BITWIDTH_PSUM * PARALLEL_MACS)
    ) mux_psum (
        .a_in (psum_data_combined_w),
        .b_in ({mult_2_o_w, mult_1_o_w}),
        .sel_i(psum_select),
        .y_o  ({adder_2_summand_2, adder_1_summand_2})
    );
  end else begin : gen_parallel_psum_multiplexer
    mux2 #(
        .DATA_WIDTH(TRANS_BITWIDTH_PSUM)
    ) mux_psum (
        .a_in (psum_data_combined_w[TRANS_BITWIDTH_PSUM-1:0]),
        .b_in ({mult_2_o_w, mult_1_o_w}),
        .sel_i(psum_select),
        .y_o  ({adder_2_summand_2, adder_1_summand_2})
    );
  end
endmodule
