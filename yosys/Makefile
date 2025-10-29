SRC_DIRS := ../hdl/
TOP_MODULE := OpenEye_FPGA
VERILOG_SRCS := $(shell find $(SRC_DIRS) -name "*.v" | sort)

TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
TRIAL_DIR := runs/trial_$(TIMESTAMP)
GEN_READ_SCRIPT := $(TRIAL_DIR)/gen_read.ys
GEN_SYNTH_SCRIPT := $(TRIAL_DIR)/gen_synth.ys
STAT_SCRIPT := $(TRIAL_DIR)/stat.ys

OUT_JSON := $(TRIAL_DIR)/$(TOP_MODULE).json
OUT_NETLIST := $(TRIAL_DIR)/netlist.v
DESIGN_JSON := $(TRIAL_DIR)/design.json

READ_LOG := $(TRIAL_DIR)/yosys_read.log
SYNTH_LOG := $(TRIAL_DIR)/yosys_synth.log
STAT_LOG := $(TRIAL_DIR)/yosys_stat.log
PNR_LOG := $(TRIAL_DIR)/nextpnr.log

CCF_FILE := olimex.ccf
PNR_OUT_TXT := $(TRIAL_DIR)/impl.txt
HIER_STAT_TXT := $(TRIAL_DIR)/hierarchy_stat.txt

.PHONY: all clean synth pnr hier_stat

all: synth pnr hier_stat

# -------------------------
# 1. Synthese mit Yosys
# -------------------------
synth: $(GEN_SYNTH_SCRIPT)
	@mkdir -p $(TRIAL_DIR)
	@echo "Starting Yosys synthesis..."
	yosys -s $(GEN_SYNTH_SCRIPT) > $(SYNTH_LOG) 2>&1
	@echo "Yosys log saved to $(SYNTH_LOG)"

$(GEN_SYNTH_SCRIPT):
	@mkdir -p $(TRIAL_DIR)
	@echo "# Combined Yosys read + synth script" > $(GEN_SYNTH_SCRIPT)
	@for f in $(VERILOG_SRCS); do \
		echo "read_verilog -sv -D SYNTHESIS $$f" >> $(GEN_SYNTH_SCRIPT); \
	done
	@echo "hierarchy -check -top $(TOP_MODULE)" >> $(GEN_SYNTH_SCRIPT)
	@echo "proc; opt; clean" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_json $(DESIGN_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "synth_gatemate \\" >> $(GEN_SYNTH_SCRIPT)
	@echo "    -top $(TOP_MODULE) \\" >> $(GEN_SYNTH_SCRIPT)
	@echo "    -luttree \\" >> $(GEN_SYNTH_SCRIPT)
	@echo "    -nomx8 \\" >> $(GEN_SYNTH_SCRIPT)
	@echo "    -nomult" >> $(GEN_SYNTH_SCRIPT)
	@echo "dfflegalize" >> $(GEN_SYNTH_SCRIPT)
	@echo "clean" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_json $(OUT_JSON)" >> $(GEN_SYNTH_SCRIPT)
	@echo "write_verilog $(OUT_NETLIST)" >> $(GEN_SYNTH_SCRIPT)
	@echo "read_json $(OUT_JSON)" > $(TRIAL_DIR)/stat.ys
	@echo "stat" >> $(TRIAL_DIR)/stat.ys
	@echo "show -format svg -prefix $(TRIAL_DIR)/circuit" >> $(TRIAL_DIR)/stat.ys

# -------------------------
# 2. Place & Route mit nextpnr
# -------------------------
pnr: $(OUT_JSON)
	@if [ ! -f $(CCF_FILE) ]; then \
		echo "No $(CCF_FILE) found — creating empty one."; \
		touch $(CCF_FILE); \
	fi
	@echo "Starting place and route with nextpnr-himbaechel..."
	nextpnr-himbaechel \
		--device=CCGM1A2 \
		--json $(OUT_JSON) \
		-o ccf=$(CCF_FILE) \
		-o out=$(PNR_OUT_TXT) \
		--router router2 \
		--vopt allow-unconstrained > $(PNR_LOG) 2>&1
	@echo "NextPNR log saved to $(PNR_LOG)"
	@echo "Implementation output: $(PNR_OUT_TXT)"

# -------------------------
# 3. Hierarchische Ressourcenanalyse
# -------------------------
hier_stat: $(DESIGN_JSON)
	@echo "Generating hierarchical resource analysis..."
	@python3 scripts/hier_stat.py $(DESIGN_JSON) $(TOP_MODULE) > $(HIER_STAT_TXT)
	@echo "Hierarchy stats written to $(HIER_STAT_TXT)"

# -------------------------
# 4. Cleanup
# -------------------------
clean:
	rm -rf runs
