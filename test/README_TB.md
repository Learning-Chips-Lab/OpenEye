# Testbench - Compilation and Execution Guide

This guide shows how to compile and run testbenches using iverilog and the generic Makefile.

## Quick Start

```bash
# Navigate to the test directory
cd test/

# Run the default testbench (PE_tb)
make

# Run a specific testbench
make TB=my_testbench
```

## Dense/GEMM split-K regression

The FPGA Dense path splits K across cluster rows and distributes consecutive
activation pairs round-robin across the PE rows within each cluster. Cluster
columns compute different output features. For K=32, two cluster rows and three
PE rows, the ramp input `1..32` is assigned as follows:

| Cluster row | PE row 0 | PE row 1 | PE row 2 |
|---|---|---|---|
| 0 | 1, 2, 7, 8, 13, 14 | 3, 4, 9, 10, 15, 16 | 5, 6, 11, 12, 17, 18 |
| 1 | 19, 20, 25, 26, 31, 32 | 21, 22, 27, 28, 0, 0 | 23, 24, 29, 30, 0, 0 |

`fc_storage_valid_i` marks a pair on the converter input. Each converter writes
only its cluster row's pairs, with one bank per PE row. During readout it holds
each pair for two clocks, matching the PE activation pipeline, and sends exactly
the configured number of activations per PE. Padding is included in the input
stream; the PE may omit zero activations when constructing its sparse scratchpad.
Both routing modes still reduce partial K results vertically.

The full-system sweep covers K×N = 4×4, 8×8, 4×8, 8×4, 16×16 and 32×32
with both MAC widths and routing modes, plus 31×8 and 63×8 padding cases.
These are matrix dimensions; the hardware array remains two cluster columns
by two cluster rows. Dense reference dot products run in-process to avoid
starting a Python interpreter for every output; their results are independently
checked against NumPy in `test/test_dense_reference.py`.

Run from the repository root:

```bash
# Fast converter checks: bank assignment, zeros, input gaps, repeated loads,
# two K tiles, and waiting for the PE cluster to become ready.
openeye_env/bin/python3 -m pytest -q test/cocotb_iact_stream_constructor/test_fc_split_k.py

# Complete DMA-to-output checks with random operands, both MAC widths and
# routing modes, plus odd K sizes. Run serially to avoid shared runner state.
OPENEYE_MAX_PROCS=2 openeye_env/bin/python3 -m pytest -q test/cocotb_fpga/test_gemm_layer.py

# A ramp identifies input positions; constant operands alone cannot detect
# activations sent to the wrong PE.
OPENEYE_MAX_PROCS=2 OPENEYE_RAMP_IACTS=1 OPENEYE_CONST_WGHTS=1 \
  openeye_env/bin/python3 -m pytest -q test/cocotb_fpga/test_gemm_layer.py
```

These system tests exercise a single Dense input vector (M=1), rather than
batched GEMM or a complete Transformer/SSM model.

## Command Line Options

### Option 1: Using the Makefile (Recommended)

The easiest way to run any testbench:

```bash
# Compile and run default testbench (PE_tb)
make

# Compile and run a specific testbench
make TB=my_testbench

# Run and open waveform viewer
make wave TB=PE_tb

# Clean generated files
make clean TB=PE_tb

# View available options
make help
```

### Option 2: Manual iverilog Command

If you prefer to run iverilog directly for a specific testbench:

```bash
# Example for PE_tb
iverilog -g2012 -Wall -Winfloop -Wno-timescale \
  -o PE_tb.vvp \
  ../hdl/PE.v \
  ../hdl/adder.v \
  ../hdl/data_pipeline.v \
  ../hdl/multiplier.v \
  ../hdl/mux2.v \
  ../hdl/mux_iact.v \
  ../hdl/SPad_DP.v \
  ../hdl/SPad_SP.v \
  ../hdl/RAM_DP.v \
  ../hdl/RAM_SP.v \
  ../hdl/RAM_DP_generic.v \
  ../hdl/RAM_DP_RW.v \
  ../hdl/RAM_DP_RW_generic.v \
  ../hdl/RAM_SP_generic.v \
  ../hdl/data_pipeline_iact.v \
  ../hdl/data_pipeline_wght.v \
  ../hdl/SPAD_DP_RW.v \
  PE_tb.v

# Run the simulation
vvp PE_tb.vvp
```

### Option 3: Simplified One-Liner

```bash
# For PE testbench
iverilog -g2012 -o PE_tb.vvp ../hdl/*.v PE_tb.v && vvp PE_tb.vvp

# For a different testbench
iverilog -g2012 -o my_testbench.vvp ../hdl/*.v my_testbench.v && vvp my_testbench.vvp
```

## Makefile Parameters

The Makefile supports the following parameters:

- `TB` - Testbench name without .v extension (default: PE_tb)
- `SIM` - Simulator to use (default: iverilog)

Examples:
```bash
make TB=PE_tb               # Run PE testbench
make TB=my_module_tb        # Run custom testbench
make wave TB=PE_tb          # Run and view waveforms
make clean TB=PE_tb         # Clean specific testbench files
```

## Command Line Flags Explained

- `-g2012` - Use SystemVerilog-2012 standard (supports modern Verilog syntax)
- `-Wall` - Enable all warnings
- `-Winfloop` - Warn about infinite loops
- `-Wno-timescale` - Suppress timescale warnings
- `-o <file>.vvp` - Specify output file name

## Viewing Waveforms

Testbenches generate VCD (Value Change Dump) files for waveform viewing.

### Using GTKWave

```bash
# Option 1: Using Makefile (for PE_tb)
make wave

# Option 2: Using Makefile (for specific testbench)
make wave TB=my_testbench

# Option 3: Manual command
gtkwave <testbench_name>.vcd &
```

### Recommended Signals to View (Example for PE Testbench)

In GTKWave, you can add relevant signals for debugging. For the PE testbench:

**Clock & Reset:**
- `clk_i`, `rst_ni`

**Input Activations:**
- `iact_data_i`, `iact_enable_i`, `iact_ready_o`

**Weights:**
- `wght_data_i`, `wght_enable_i`, `wght_ready_o`

**Partial Sums:**
- `psum_data_i`, `psum_enable_i`, `psum_data_o`, `psum_enable_o`

**Control:**
- `compute_i`

**Internal Signals:**
- `dut.current_state_computing` - FSM state
- `dut.adder_1.sum_o`, `dut.adder_2.sum_o` - Adder outputs

## Expected Output

When a testbench runs successfully, you should see test-specific output. For example, the PE testbench shows:

```
================================================================================
PE Convolution Test
================================================================================
[Configuration and test details...]
================================================================================
Result Validation
================================================================================
PASS: Output[0] = 25 (expected 25)
*** CONVOLUTION TEST PASSED ***
================================================================================
Test Summary
================================================================================
Total Errors: 0
*** ALL TESTS PASSED ***
================================================================================
```

Refer to each testbench's specific documentation for expected output format.

## Troubleshooting

### Error: "Can't find file"

Make sure you're in the `test/` directory:
```bash
cd test/
```

### Error: "syntax error" or "module not found"

Check that all required HDL files exist:
```bash
ls -la ../hdl/*.v
ls -la <testbench_name>.v
```

### Error: "testbench file not found"

Ensure the testbench file exists and matches the TB parameter:
```bash
# If running: make TB=my_testbench
# The file should be: test/my_testbench.v
ls -la my_testbench.v
```

### Simulation Timeout

If the simulation times out, check:
1. The DUT (Device Under Test) is correctly instantiated
2. Clock is running (view in GTKWave)
3. Reset is properly released
4. Input stimulus is being applied correctly

### No Waveform File Generated

Verify the VCD dump is enabled in your testbench:
```verilog
initial begin
    $dumpfile("<testbench_name>.vcd");
    $dumpvars(0, <module_instance>);
end
```

Or if using a parameter:
```verilog
parameter CREATE_VCD = 1;
```

## File Locations

**Generic Structure:**
- **Testbench**: `test/<testbench_name>.v`
- **HDL Sources**: `hdl/*.v`
- **Makefile**: `test/Makefile`
- **VCD Output**: `test/<testbench_name>.vcd`
- **Compiled Binary**: `test/<testbench_name>.vvp`

**Example (PE Testbench):**
- **Testbench**: `test/PE_tb.v`
- **Main Module**: `hdl/PE.v`
- **VCD Output**: `test/PE_tb.vcd`
- **Compiled Binary**: `test/PE_tb.vvp`

## Advanced Usage

### Running with Custom Parameters

Most testbenches allow you to modify test parameters directly in the source file:

```verilog
// Edit your testbench file to change parameters
parameter DATA_WIDTH = 8;
parameter ADDR_WIDTH = 10;
// ... etc
```

### Adding More Test Cases

You can add additional test cases by:
1. Duplicating test sequences in the `initial` block
2. Creating task/function calls for repeated test patterns
3. Using `$readmemh` to load test vectors from files

### Creating a New Testbench

To create a new testbench that works with this Makefile:

1. Create your testbench file: `test/my_module_tb.v`
2. Include VCD dump generation:
   ```verilog
   initial begin
       $dumpfile("my_module_tb.vcd");
       $dumpvars(0, my_module_tb);
   end
   ```
3. Update the `SOURCES` variable in the Makefile if needed
4. Run with: `make TB=my_module_tb`

## Available Testbenches

Current testbenches in this directory:
- `PE_tb` - Processing Element testbench (default)
- *(Add other testbenches here as they are created)*

## Comparison: Verilog vs CocoTB

| Feature | Verilog TB | CocoTB TB |
|---------|------------|-----------|
| Language | Verilog | Python |
| Setup | Simple, direct | Requires cocotb |
| Test Cases | Manual implementation | Parametric, automated |
| Debugging | GTKWave waveforms | Python prints + waveforms |
| Complexity | Medium | High (full protocol) |
| Use Case | Basic verification | Comprehensive testing |

For comprehensive testing, consider using CocoTB testbenches where available.

## Debug switches (environment variables)

All opt-in unless noted. They exist because a bare pass/fail says nothing about
*why* a result is wrong; each one turns a mismatch into a measurement.

### FPGA top level (`test/cocotb_fpga/`)

| Variable | Effect |
|---|---|
| `OPENEYE_PROBE_FSM=1` | Log every main-FSM transition with the decoded config fields that steer it. |
| `OPENEYE_FAIL_ON_STALL=<cycles>` | Fail once neither FSM advances for that many cycles, instead of running to the 15 ms sim timeout (hours of wall clock). The report names the FSM states, the psum handshake vectors, the router modes and the per-PE iact/weight counts. Needs `OPENEYE_PROBE_FSM`. |
| `OPENEYE_ZERO_IACTS=1` | Zero layer 0's input, so every output must equal its bias alone. Separates the bias/psum path from iact and weight delivery. |
| `OPENEYE_CONST_IACTS=<v>` | Force every layer-0 activation to a constant. |
| `OPENEYE_CONST_WGHTS=<v>` | Force every weight to a constant. With both at 1 each output becomes a *count* of the products that actually accumulated - this is how "gemm is wrong" became "gemm accumulates 9 of 32 products". **Always sweep at least two values**: a single constant cannot distinguish a data-independent DUT from a degenerate reference. |
| `DUMP_CLUSTER_IACT=1` | Per cluster, count iact handshakes at three hops (`ext`, `glb`, `pe`) plus how many carried non-zero payload; per PE, the selected lane with its valid and accepted counts. This is what showed activations arriving at every cluster and only some carrying data. |
| `DUMP_PSUM_BUFFERS=1` | End-of-run dump: per-PE SPAD occupancy, psum buffers, iact path, and the reports above. |
| `OPENEYE_MAX_PROCS=<n>` | Cap the helper processes the reference calculation spawns. Without it one pytest worker started 37 processes and drove an 8-core machine to load 94; at `3` it is 7 processes and load ~12. |

### PE cluster (`test/cocotb_PE_cluster/`)

| Variable | Effect |
|---|---|
| `OPENEYE_PSUM_TERMS=1` | On a psum mismatch, print every product feeding that psum as `(weight, iact, product, pe_y)`, and flag any single term or operand mis-pairing that accounts for the difference. |
| *(always on)* | The SPAD encoder reports any zero run that does not fit its overhead field. An undecodable weight stream would make every downstream psum comparison meaningless, so this is checked before the terms are interpreted. |

## Focused regression tests

The big parametrised suites are too large to run whole (`test_PE_CLUSTER.py`
alone collects 32256 cases), so these hold everything fixed but the one axis
that matters:

| Test | Purpose | Runtime |
|---|---|---|
| `test/cocotb_PE_cluster/test_PE_CLUSTER_sparse.py` | Weight-sparsity sweep at one shape and seed. Turns the sparse failures into a monotone curve: `PARALLEL_MACS=2` passes to 30 % and fails from 40 %. | ~15 s, 14 cases |
| `test/cocotb_fpga/test_conv_const.py::test_conv_const_single_layer` | One conv layer with constant operands, so each output is a product count. Compute and read-out only, no interlayer step. | ~65 s per case, 6 cases |
| `test/cocotb_fpga/test_conv_const.py::test_conv_const_two_layers` | Two stacked conv layers (`LAYER=Convolution_Stack`): the only focused test of the interlayer psum->iact write-back, with no pooling layer in between - the MNIST net cannot separate the two. Sweeps c = 1 and 8, because 1 and 2 quantise to the same interlayer byte. | ~65 s per case, 8 cases |

## References

- Generic Makefile: `test/Makefile`
- HDL Sources: `hdl/`
- CocoTB Tests: `test/cocotb_*/` (if available)
- Project Documentation: `doc/`
