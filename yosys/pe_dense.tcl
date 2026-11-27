yosys -import

set run_dir $::env(SYNTH_RUN_DIR)

read_verilog -sv -DSYNTHESIS -DNO_TRACE -DUSE_INTERNAL_PARAMS_PE -Ihdl \
    hdl/PE.v \
    hdl/adder.v \
    hdl/adder_tree.v \
    hdl/data_pipeline_wght.v \
    hdl/data_pipeline_iact.v \
    hdl/dsp_unit.v \
    hdl/multiplier.v \
    hdl/mux2.v \
    hdl/mux_iact.v \
    hdl/SPad_DP.v \
    hdl/SPad_DP_RW.v \
    hdl/SPad_SP.v \
    hdl/RAM_DP.v \
    hdl/RAM_DP_RW.v \
    hdl/RAM_DP_RW_generic.v \
    hdl/RAM_SP.v \
    hdl/RAM_DP_generic.v \
    hdl/RAM_SP_generic.v

chparam -set PARALLEL_MACS 1 \
    -set SPARSITY_EN 0 \
    -set SERIAL 1 PE

hierarchy -check -top PE
yosys rename -top PE

synth_gatemate -top PE -luttree -nomx8 -noiopad -noclkbuf

check -assert

tee -o $run_dir/resources.json stat -json -top PE
write_json $run_dir/netlist.json