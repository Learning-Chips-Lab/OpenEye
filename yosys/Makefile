SRC_DIRS := ../hdl/
TOP_MODULE := OpenEye_FPGA
VERILOG_SRCS := $(shell find $(SRC_DIRS) -name "*.v" | sort)
TARGET_FPGA := CCGM1A1

ifndef TIMESTAMP
    TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
endif
TRIAL_DIR := runs/trial_$(TIMESTAMP)
GEN_SYNTH_SCRIPT := $(TRIAL_DIR)/gen_synth.ys

OUT_JSON := $(TRIAL_DIR)/$(TOP_MODULE).json
PNR_OUT_TXT := $(TRIAL_DIR)/impl.txt
PNR_LOG := $(TRIAL_DIR)/nextpnr.log
CCF_FILE := olimex.ccf
HIER_STAT_TXT := $(TRIAL_DIR)/hierarchy_stat.txt

.PHONY: all clean synth hier_stat pnr

all: synth hier_stat pnr 

hier_stat: synth
	@echo "Generating hierarchical resource analysis..."
	@python3 scripts/hier_stat.py $(OUT_JSON) $(TOP_MODULE) > $(HIER_STAT_TXT)
	@echo "Hierarchy stats successfully written to: $(HIER_STAT_TXT)"

synth: $(GEN_SYNTH_SCRIPT)
	@mkdir -p $(TRIAL_DIR)
	@echo "Starting minimalist Yosys synthesis..."
	yosys -s $(GEN_SYNTH_SCRIPT)
	@echo "Synthesis finished successfully."

$(GEN_SYNTH_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Minimalist Yosys Script" > $(GEN_SYNTH_SCRIPT)
	@$(foreach f,$(VERILOG_SRCS),echo "read_verilog -sv -D SYNTHESIS -D NO_TRACE $(f)" >> $(GEN_SYNTH_SCRIPT);)
	@echo "hierarchy -check -top $(TOP_MODULE)" >> $(GEN_SYNTH_SCRIPT)
	@echo "synth_gatemate -top $(TOP_MODULE) -luttree -nomx8" >> $(GEN_SYNTH_SCRIPT)
	@echo "stat -top $(TOP_MODULE) -json" >> $(GEN_SYNTH_SCRIPT)
	
	@echo "write_json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)

pnr: synth
	@echo "Starting minimalist Place & Route..."
	# NEU: Leitet die NextPNR-Ausgabe wieder sauber in das nextpnr.log um
	nextpnr-himbaechel \
		--device=$(TARGET_FPGA) \
		--json $(OUT_JSON) \
		-o ccf=$(CCF_FILE) \
		-o out=$(PNR_OUT_TXT) \
		--freq 40 \
		--router router2 \
		--placer-heap-cell-placement-timeout 0 \
		-o allow-unconstrained > $(PNR_LOG) 2>&1
	@echo "Place & Route finished."

clean:
	rm -rf runs