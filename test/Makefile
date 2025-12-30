# Makefile for Verilog Testbenches
# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1

# Simulator selection
SIM ?= iverilog

# Testbench selection (can be overridden via command line)
TB ?= PE_tb

# Directories
HDL_DIR = ../hdl
TB_DIR = .

# Source files - PE and all its dependencies
SOURCES = \
	$(HDL_DIR)/PE.v \
	$(HDL_DIR)/adder.v \
	$(HDL_DIR)/data_pipeline.v \
	$(HDL_DIR)/data_pipeline_iact.v \
	$(HDL_DIR)/data_pipeline_wght.v \
	$(HDL_DIR)/multiplier.v \
	$(HDL_DIR)/mux2.v \
	$(HDL_DIR)/mux_iact.v \
	$(HDL_DIR)/SPad_DP.v \
	$(HDL_DIR)/SPad_SP.v \
	$(HDL_DIR)/RAM_DP.v \
	$(HDL_DIR)/RAM_SP.v \
	$(HDL_DIR)/RAM_DP_generic.v \
	$(HDL_DIR)/RAM_DP_RW.v \
	$(HDL_DIR)/RAM_DP_RW_generic.v \
	$(HDL_DIR)/RAM_SP_generic.v \
	$(HDL_DIR)/SPAD_DP_RW.v

# Testbench file (derived from TB parameter)
TB_FILE = $(TB).v

# Output files (derived from TB parameter)
VVP_FILE = $(TB).vvp
VCD_FILE = $(TB).vcd

# Compiler flags
IVERILOG_FLAGS = -g2012 -Wall -Winfloop -Wno-timescale

# Default target
.PHONY: all
all: compile run

# Compile the testbench
.PHONY: compile
compile:
	@echo "Compiling $(TB) testbench..."
	$(SIM) $(IVERILOG_FLAGS) -o $(VVP_FILE) $(SOURCES) $(TB_FILE)
	@echo "Compilation complete!"

# Run the simulation
.PHONY: run
run: compile
	@echo ""
	@echo "Running $(TB) simulation..."
	@echo "========================================"
	vvp $(VVP_FILE)
	@echo "========================================"
	@echo "Simulation complete!"
	@echo "VCD waveform file: $(VCD_FILE)"

# View waveforms with GTKWave
.PHONY: wave
wave: run
	@echo "Opening waveform viewer..."
	gtkwave $(VCD_FILE) &

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f $(VVP_FILE) $(VCD_FILE)
	@echo "Clean complete!"

# Help target
.PHONY: help
help:
	@echo "Generic Testbench Makefile"
	@echo "=========================="
	@echo ""
	@echo "Targets:"
	@echo "  all      - Compile and run the testbench (default)"
	@echo "  compile  - Compile the testbench only"
	@echo "  run      - Compile and run the simulation"
	@echo "  wave     - Run simulation and open GTKWave"
	@echo "  clean    - Remove generated files"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Parameters:"
	@echo "  TB       - Testbench name without .v extension (default: PE_tb)"
	@echo "  SIM      - Simulator to use (default: iverilog)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Compile and run PE_tb (default)"
	@echo "  make TB=my_testbench    # Compile and run my_testbench.v"
	@echo "  make wave TB=PE_tb      # Run PE_tb and view waveforms"
	@echo "  make clean TB=PE_tb     # Clean PE_tb generated files"
