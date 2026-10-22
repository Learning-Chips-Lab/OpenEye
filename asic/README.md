# OpenEye ASIC — IHP SG13G2

Backend flow (RTL → GDS) for the OpenEye accelerator on IHP SG13G2, built on
the [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools) container
(LibreLane 3.x, OpenROAD, Yosys, Magic, KLayout, Netgen).

## Layout

```
asic/
├── README.md               this file
├── .gitignore              runs/ (flow outputs) is not committed
├── rtl/
│   ├── sg13g2_io_blackbox.v  clean pad blackbox models (OpenSTA-compatible)
│   ├── bondpad_70x70.lef     bondpad master (PDK does not ship one, TODO in PDK)
│   ├── bondpad_70x70.gds     bondpad geometry (Metal3..TopMetal2)
│   └── generated/            (gitignored) dma_storage.v, parameters_*.vh
├── scripts/
│   ├── container.sh          start/stop/status/exec of the worker container
│   ├── env.sh                PDK env (ihp-sg13g2, sg13g2_stdcell, sg13g2_io)
│   └── run_smoke.sh          counter smoke test through the full Chip flow
├── smoke/counter/            smoke design (RTL + LibreLane config + SDC)
└── runs/                     (gitignored) LibreLane run output
```

## Environment

- Container: `hpretl/iic-osic-tools:latest` (image built 2026-10-15, tag 2026.10)
- LibreLane v3.1.0.dev3, OpenROAD 26Q4, Yosys 0.70, KLayout 0.30.5, Magic 8.3.589
- PDK: `ihp-sg13g2` in-container at `/foss/pdks` (PDK commit `a49358cc`),
  30 SRAM macro variants (1P/2P), `sg13g2_io` pad library (22 cells),
  3 timing corners (typ 1.2 V/25 °C, fast 1.32 V/−40 °C, slow 1.08 V/125 °C)
- Worker container: `iic-osic-tools_asic` (headless, uid 1000),
  mounts `/home/utz/projekte` → `/foss/designs`
  (repo therefore at `/foss/designs/OpenEye`)

### Usage

```bash
asic/scripts/container.sh start          # headless worker container
asic/scripts/container.sh exec "librelane --version"
asic/scripts/run_smoke.sh <runname>      # full Chip-flow smoke test
asic/scripts/container.sh stop
```

The user's interactive container (`start_x.sh`) keeps running independently.

## Phase 0 — Flow smoke test (2026-10-22)

Reference runs (kept in `runs/`):

- `smoke_sg13g2_phase0` — classic flow, 32-bit counter (no pads):
  **full success**, clean STA/DRC/LVS, GDS out.
- `smoke_sg13g2_phase0_chip17` — **Chip flow** (padring + sealring + filler +
  density), 8-bit counter with 16 explicit `sg13g2_io` pads + bondpads:
  all 78 steps executed, GDS out (34 MB), timing met (WNS/TNS = 0 in all
  corners), but with PDK integration gaps in sign-off (below).

### What the Chip-flow run validated

1. Yosys synthesis with pad blackboxes: Liberty blackboxes do **not** expose
   power pins as connectable ports → pads are blackboxed via
   `EXTRA_VERILOG_MODELS` with clean Verilog models
   (`rtl/sg13g2_io_blackbox.v`; the PDK's `sg13g2_io.v` is not parseable by
   OpenSTA's netlist reader — specify blocks).
2. Absolute floorplan: `FP_SIZING: absolute` + `DIE_AREA`; core margins as
   multiples of site width/height (`LEFT/RIGHT_MARGIN_MULT=667` → 320.16 µm,
   `TOP/BOTTOM_MARGIN_MULT=85` → 321.3 µm) so the core does not overlap the
   pad band (140+180 µm).
3. Padring (`meta.flow = "Chip"`): pads listed per side in
   `PAD_SOUTH/EAST/NORTH/WEST`, `PAD_SPACING_MULTIPLE=80` (pad site is
   1 µm wide, pads span 80 sites), corners + fillers auto-placed, bondpads
   placed from the custom master.
4. Bondpad master: OpenROAD 26Q4 LEF parser quirks —
   - MACRO must be closed with `END <macro_name>` (IHP PDK convention),
     `END MACRO` is rejected (LEFPARS-1634),
   - `BEGIN GROUP` is rejected by the parser → geometry-free LEF master
     (IHP PDK convention: LEF without GROUP, geometry in GDS),
   - `place_bondpad` requires `CLASS COVER` and a determinable top layer
     (PIN with PORT on TopMetal2).
5. GDS merge: bondpad GDS via `EXTRA_GDS` (loaded by magic and KLayout
   streamout).
6. Timing: 20 ns clock met in all three corners (setup/hold WNS = 0).

### Open PDK integration gaps (must be solved before tape-out)

These are **not flow bugs**; the IHP open PDK marks the pad power/bondpad
area as work-in-progress ("TODO bondpads need to be part of the PDK").

1. **Bondpad ↔ pad port not electrically connected** (LVS: 15 net mismatches,
   e.g. `out_pad[0]`: layout has `bondpad_70x70/pad`, schematic has
   `sg13g2_IOPadOut30mA/pad`). The guessed bondpad GDS (solid 70×70 on
   Metal3..TopMetal2) does not merge with the pad's `pad`-pin PORT metal.
   Also: west-side pads are placed with the `pad` port facing the core side
   while the bondpad sits die-side (orientation/offset inconsistency).
   → Need the proper IHP bondpad design (layers, vias, orientation per side)
   or confirmation from IHP how pad→bondpad connectivity is intended.
2. **Pad `vdd` pins not connected to the power grid** (PSM-0069 in
   `check_power_grid`; 44 pins + 4 bondpad pins). `vss/iovdd/iovss` connect,
   `vdd` does not — none of the tested ring variants
   (`-core_offsets`, `-pad_offsets`, `-connect_to_pads`, `-nets`,
   `-extend_to_boundary`) closes it. `RUN_IRDROP_REPORT` was disabled for
   the smoke test. → Pad power architecture needs a PDK-level definition.
3. **Sealring offset vs. bondpads** (1.59 M × `Seal.l`: "no structures
   outside sealring boundary"): sealring at 140 µm, bondpads at 70..140 µm
   (outside). → Sealring offset must move inside the bondpad band
   (KLAYOUT sealring step parameters / PDK config).
4. `check_power_grid` in `generatepdn` is a warning; the fatal check sits in
   `IRDropReport` — re-enable once gap 2 is resolved.

### Config notes carried over to the OpenEye design (Phase 2)

- `meta.flow = "Chip"`, `FP_SIZING: absolute` + `DIE_AREA` (and core margins)
- `EXTRA_VERILOG_MODELS` (pad blackboxes), `EXTRA_LEFS` + `EXTRA_GDS`
  (bondpad master)
- `PAD_SOUTH/EAST/NORTH/WEST` lists, `PAD_SPACING_MULTIPLE`
- SRAM macros: `MACROS` config (gds/lef/lib per corner) +
  `CELL_LIBS` extended per corner (dict replace semantics) +
  `EXTRA_VERILOG_MODELS` for macro blackboxing during synthesis
- PDK defaults: VDD 1.2 V (net VPWR), no welltap/endcap cells,
  CTS root `sg13g2_buf_8`, routing Metal2..TopMetal2
