SRC_DIRS := ../hdl/
TOP_MODULE := OpenEye_FPGA
VERILOG_SRCS := $(shell find $(SRC_DIRS) -name "*.v" | sort)

TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
TRIAL_DIR := runs/trial_$(TIMESTAMP)
GEN_READ_SCRIPT := $(TRIAL_DIR)/gen_read.ys
GEN_SYNTH_SCRIPT := $(TRIAL_DIR)/gen_synth.ys
STAT_SCRIPT := $(TRIAL_DIR)/stat.ys

OUT_JSON := $(TRIAL_DIR)/$(TOP_MODULE).json

READ_LOG := $(TRIAL_DIR)/yosys_read.log
SYNTH_LOG := $(TRIAL_DIR)/yosys_synth.log
STAT_LOG := $(TRIAL_DIR)/yosys_stat.log

.PHONY: all clean

all: $(GEN_SYNTH_SCRIPT)
	@mkdir -p $(TRIAL_DIR)
	yosys -s $(GEN_SYNTH_SCRIPT) > $(SYNTH_LOG) 2>&1
	@echo "Yosys log saved to $(SYNTH_LOG)"

$(GEN_READ_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Yosys read script" > $(GEN_READ_SCRIPT)
	@for f in $(VERILOG_SRCS); do \
		echo "read_verilog -D SYNTHESIS $$f" >> $(GEN_READ_SCRIPT); \
	done

$(GEN_SYNTH_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Combined Yosys read + synth script" > $(GEN_SYNTH_SCRIPT)
	@for f in $(VERILOG_SRCS); do \
		echo "read_verilog -D SYNTHESIS $$f" >> $(GEN_SYNTH_SCRIPT); \
	done
	@echo "hierarchy -top $(TOP_MODULE)" >> $(GEN_SYNTH_SCRIPT)
	@echo "synth_gatemate -top $(TOP_MODULE) -json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_verilog $(TRIAL_DIR)/netlist.v" >> $(GEN_SYNTH_SCRIPT)
	@echo "read_json $(OUT_JSON)" > $(TRIAL_DIR)/stat.ys
	@echo "stat" >> $(TRIAL_DIR)/stat.ys
	@echo "show -format svg -prefix $(TRIAL_DIR)/circuit" >> $(TRIAL_DIR)/stat.ys

clean:
	rm -rf runs
