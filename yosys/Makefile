SRC_DIRS := ../hdl/
TOP_MODULE := OpenEye_FPGA
VERILOG_SRCS := $(shell find $(SRC_DIRS) -name "*.v" | sort)

TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
TRIAL_DIR := runs/trial_$(TIMESTAMP)
GEN_READ_SCRIPT := $(TRIAL_DIR)/gen_read.ys
GEN_SYNTH_SCRIPT := $(TRIAL_DIR)/gen_synth.ys
OUT_JSON := $(TRIAL_DIR)/$(TOP_MODULE).json

READ_LOG := $(TRIAL_DIR)/yosys_read.log
SYNTH_LOG := $(TRIAL_DIR)/yosys_synth.log
STAT_LOG := $(TRIAL_DIR)/yosys_stat.log

all: $(GEN_READ_SCRIPT) $(GEN_SYNTH_SCRIPT)
	@mkdir -p $(TRIAL_DIR)
	yosys -c $(GEN_READ_SCRIPT) > $(READ_LOG) 2>&1
	@echo "Read log saved to $(READ_LOG)"
	yosys -c $(GEN_SYNTH_SCRIPT) > $(SYNTH_LOG) 2>&1
	@echo "Synth log saved to $(SYNTH_LOG)"
	yosys -c $(TRIAL_DIR)/stat.ys > $(STAT_LOG) 2>&1
	@echo "Stat log saved to $(STAT_LOG)"

$(GEN_READ_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Yosys read script" > $(GEN_READ_SCRIPT)
	@for f in $(VERILOG_SRCS); do \
		echo "read_verilog -D SYNTHESIS $$f" >> $(GEN_READ_SCRIPT); \
	done

$(GEN_SYNTH_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Yosys synth script" > $(GEN_SYNTH_SCRIPT)
	@echo "hierarchy -top $(TOP_MODULE)" >> $(GEN_SYNTH_SCRIPT)
	@echo "synth_gatemate -top $(TOP_MODULE) -json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_verilog $(TRIAL_DIR)/netlist.v" >> $(GEN_SYNTH_SCRIPT)
	@echo "# Generate stat script" > $(TRIAL_DIR)/stat.ys
	@echo "read_json $(OUT_JSON)" >> $(TRIAL_DIR)/stat.ys
	@echo "stat" >> $(TRIAL_DIR)/stat.ys
	@echo "show -format svg -prefix $(TRIAL_DIR)/circuit" >> $(TRIAL_DIR)/stat.ys

clean:
	rm -rf runs