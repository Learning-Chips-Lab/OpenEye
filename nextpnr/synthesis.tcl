# Synthesis script for PE.v and its hierarchy using Yosys
yosys -import

# Read all Verilog files in the hierarchy
read_verilog ../hdl/PE.v
read_verilog ../hdl/adder.v
read_verilog ../hdl/data_pipeline.v
read_verilog ../hdl/data_pipeline_iact.v
read_verilog ../hdl/data_pipeline_wght.v
read_verilog ../hdl/multiplier.v
read_verilog ../hdl/mux2.v
read_verilog ../hdl/mux_iact.v
read_verilog ../hdl/SPad_DP.v
read_verilog ../hdl/SPad_SP.v
read_verilog ../hdl/RAM_DP.v
read_verilog ../hdl/RAM_SP.v
read_verilog ../hdl/RAM_DP_generic.v
read_verilog ../hdl/RAM_DP_RW.v
read_verilog ../hdl/RAM_DP_RW_generic.v
read_verilog ../hdl/RAM_SP_generic.v
read_verilog ../hdl/SPAD_DP_RW.v

# Set top module
hierarchy -top PE

# Perform synthesis
synth -top PE

# Write output
write_verilog synth_PE.v
write_json PE.json