# Pflichtargumente
set xpr_path     [lindex $argv 0]
set bd_path      [lindex $argv 1]
set ip_name      [lindex $argv 2]
set report_dir   [lindex $argv 3]
set local_ip     [lindex $argv 4]

# 1. Projekt öffnen
open_project $xpr_path

# 2. IP-Repository radikal neu setzen
set_property ip_repo_paths "" [current_project]
update_ip_catalog
set abs_path [file normalize $local_ip]
set_property ip_repo_paths [list $abs_path] [current_project]
update_ip_catalog
# --- DIAGNOSE START ---
puts "\n------------------------------------------------------------"
puts "DIAGNOSE: IP-KATALOG"
set all_custom_ips [get_ipdefs -all]
foreach ipdef $all_custom_ips {
    if {[regexp {open_eye} $ipdef]} {
        puts "Gefundene Definition: $ipdef"
    }
}

puts "\nDIAGNOSE: BLOCK DESIGN CELLS"
open_bd_design $bd_path
set bd_cells [get_bd_cells -hierarchical]
foreach cell $bd_cells {
    set vlnv [get_property VLNV $cell]
    set name [get_property NAME $cell]
    puts "Zelle: $name | VLNV: $vlnv"
}
puts "------------------------------------------------------------\n"
# --- DIAGNOSE ENDE ---
# 3. VERIFIKATION: Ist die Definition im Katalog vorhanden?
set ip_defs [get_ipdefs -all -filter "VLNV =~ *open_eye*"]
if {$ip_defs eq ""} {
    puts "FATAL ERROR: IP Definition für 'open_eye' wurde im Pfad $abs_path nicht gefunden!"
    puts "Inhalt des Verzeichnisses laut TCL: [glob -nocomplain [file join $abs_path *]]"
    exit 1
}
puts "INFO: IP-Definition gefunden: $ip_defs"

# 4. Block Design öffnen
open_bd_design $bd_path

# 5. IP-Instanz finden
set ip_inst [get_bd_cells -hierarchical -filter {VLNV =~ "*open_eye*"}]
if {$ip_inst eq ""} {
    puts "FATAL ERROR: IP-Instanz im Block Design nicht gefunden!"
    exit 1
}

# 6. Lock aufheben durch Upgrade
# Wir holen das IP-Objekt der Instanz
set ip_obj [get_ips [get_property CONFIG.Component_Name $ip_inst]]
if {[get_property IS_LOCKED $ip_obj]} {
    puts "INFO: IP ist gelockt. Führe Upgrade durch..."
    upgrade_ip $ip_obj
}

# 7. Parameter setzen (Jetzt sollte es klappen!)
for {set i 5} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    if {[regexp {([^=]+)=(.+)} $arg -> key val]} {
        puts "Setting parameter $key to $val"
        if {[catch {set_property CONFIG.$key $val $ip_inst} err]} {
             puts "WARNING: Konnte Parameter $key nicht setzen: $err"
        }
    }
}

# 8. Speichern und Generieren (Force)
validate_bd_design
save_bd_design
generate_target all [get_files $bd_path] -force
export_ip_user_files -of_objects [get_files $bd_path] -no_script -force

# Create report directory
file mkdir $report_dir

# Reset and run synthesis
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}
# Reset and run implementation
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1

# Reports
report_utilization -file "$report_dir/utilization.xml" -format xml

close_project

