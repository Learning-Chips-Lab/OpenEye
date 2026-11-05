# Layout snapshots of the routed design with the OpenROAD GUI (offscreen).
#   QT_QPA_PLATFORM=offscreen ODB=<odb> OUT_DIR=<dir> \
#     openroad -gui -no_splash -exit asic/scripts/doc/layout_snapshot.tcl
# (inside the container). Writes layout_full.png (whole die, all layers) and
# layout_zoom_pe.png (a 600 um window in the core).
read_db $::env(ODB)
set out $::env(OUT_DIR)

# whole chip: routing + instances, no fill / no pin markers
gui::set_display_controls "Instances/Physical/*" visible false
gui::set_display_controls "Misc/Instances/*" visible false
save_image -area {0 0 10500 10500} -width 2000 $out/layout_full.png
# same view without signal/clock routing (placement + PDN + macros + pads)
gui::set_display_controls "Nets/Signal" visible false
gui::set_display_controls "Nets/Clock" visible false
save_image -area {0 0 10500 10500} -width 2000 $out/layout_full_noroute.png
gui::set_display_controls "Nets/Signal" visible true
gui::set_display_controls "Nets/Clock" visible true

# zoom: 600 um x 600 um around the die centre (PE array, routing visible)
# die is 10500 x 10500 um (DIEAREA in the DEF)
set cx 5250.0
set cy 5250.0
save_image -area [list [expr {$cx - 300}] [expr {$cy - 300}] [expr {$cx + 300}] [expr {$cy + 300}]] \
    -width 2400 $out/layout_zoom_center.png

# zoom on a macro pin edge (bottom row, pins face the core)
save_image -area {3000 600 3800 1400} -width 2400 $out/layout_zoom_macro_edge.png
puts "snapshots done"
