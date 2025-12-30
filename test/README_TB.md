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

## References

- Generic Makefile: `test/Makefile`
- HDL Sources: `hdl/`
- CocoTB Tests: `test/cocotb_*/` (if available)
- Project Documentation: `doc/`
