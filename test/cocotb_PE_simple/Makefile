SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(abspath $(PWD)/../../hdl/PE_simple.v)

TOPLEVEL = PE_simple
MODULE   = PE_simple_tb

export IVERILOG_DUMPER = fst

export CLOCK_LEN         = 10
export CLOCK_UNIT        = ns
export CLOCK_DELAY_INPUT = 100
export CLOCK_DELAY_UNIT_INPUT  = ps
export CLOCK_DELAY_OUTPUT = 100
export CLOCK_DELAY_UNIT_OUTPUT = ps

# Test dimensions
export B    = 4   # spatial window (iact per channel)
export C0   = 3   # input channels
export M0   = 12  # output filters
export SEED = 42

export
run:
	make -f $(shell cocotb-config --makefiles)/Makefile.sim
