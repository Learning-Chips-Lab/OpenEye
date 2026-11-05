# Per-endpoint timing + per-instance power dump for the module-level design
# report (asic/scripts/doc/module_report.py). Loads the routed design exactly
# like LibreLane's STAPostPNR (same libs per corner, netlist of the fill step,
# write_sdc output, RCX SPEF), then writes plain-text tables.
#
# Run inside the container with OpenSTA:
#   CORNER=typ RUN_DIR=<.../runs/p2_16> OUT_DIR=<dir> [DO_POWER=1] \
#     sta -no_splash -exit asic/scripts/doc/module_sta.tcl
#
# Outputs in OUT_DIR:
#   endpoints_<corner>_max.txt / _min.txt : report_checks -format end, one
#                                           line per endpoint (setup / hold)
#   power_<corner>.txt                    : inst internal switching leakage total [W]
#   clock_<corner>.txt                    : clock latency/skew summary

set corner $::env(CORNER)
set run    $::env(RUN_DIR)
set out    $::env(OUT_DIR)

set pdk /foss/pdks/ihp-sg13g2/libs.ref
switch $corner {
    typ  { set sc typ_1p20V_25C;   set io typ_1p2V_3p3V_25C;   set sr typ_1p20V_25C }
    slow { set sc slow_1p08V_125C; set io slow_1p08V_3p0V_125C; set sr slow_1p08V_125C }
    fast { set sc fast_1p32V_m40C; set io fast_1p32V_3p6V_m40C; set sr fast_1p32V_m55C }
    default { error "CORNER must be typ|slow|fast" }
}

read_liberty $pdk/sg13g2_stdcell/lib/sg13g2_stdcell_${sc}.lib.gz
read_liberty $pdk/sg13g2_io/lib/sg13g2_io_${io}.lib.gz
foreach m {1024x64_c2 1024x16_c2 1024x8_c2 2048x64_c2 2048x32_c2 4096x16_c3 4096x8_c3} {
    read_liberty $pdk/sg13g2_sram/lib/RM_IHPSG13_1P_${m}_bm_bist_${sr}.lib
}
read_verilog $run/55-openroad-fillinsertion/OpenEye_Asic.nl.v
link_design OpenEye_Asic
read_sdc $run/55-openroad-fillinsertion/OpenEye_Asic.sdc
read_spef $run/57-openroad-rcx/nom/OpenEye_Asic.nom.spef

set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um

proc stamp {msg} { puts "[exec date +%T] $msg"; flush stdout }

stamp "timing $corner"
# one line per endpoint (worst path per endpoint), all endpoints
foreach {kind tag} {max max min min} {
    sta::redirect_file_begin $out/endpoints_${corner}_${tag}.txt
    report_checks -path_delay $kind -group_path_count 3000000 \
        -endpoint_path_count 1 -format end -digits 4
    sta::redirect_file_end
    stamp "endpoints $tag done"
}
sta::redirect_file_begin $out/clock_${corner}.txt
report_clock_skew
report_clock_min_period
sta::redirect_file_end

if { [info exists ::env(DO_POWER)] && $::env(DO_POWER) } {
    stamp "power $corner"
    set sc [sta::cmd_scene]
    set f [open $out/power_${corner}.txt w]
    set n 0
    foreach inst [get_cells *] {
        set cell [get_property $inst ref_name]
        if { [string match "sg13g2_fill*" $cell] || [string match "sg13g2_decap*" $cell] } { continue }
        set p [sta::instance_power $inst $sc]
        if { [lindex $p 3] > 0 } {
            puts $f "[get_full_name $inst] $cell [join $p { }]"
        }
        incr n
    }
    close $f
    sta::redirect_file_begin $out/power_${corner}_summary.txt
    report_power
    sta::redirect_file_end
    stamp "power done ($n instances)"
}
stamp "all done"
