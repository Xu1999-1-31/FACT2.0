set starttime [clock seconds]
echo "INFORM: Start job at: " [clock format $starttime -gmt false]
set is_si_enabled false

set top_design aes_cipher_top

set link_library "* ../Timing_Lib/tcbn28hpcplusbwp7t40p140ssg0p72v125c_ccs.db"

set netlist "../Icc2Output/${top_design}_eco.v"
set sdc "../Icc2Output/${top_design}_eco.sdc"
set spef "../Icc2Output/${top_design}_eco.rcworst_125.spef"

source -e -v ../pt_variable.tcl

set NET_FILE $netlist 
set SDC_FILE $sdc
set SPEF_FILE $spef

set_app_var read_parasitics_load_locations true
read_verilog  $NET_FILE
link

read_parasitics -keep_capacitive_coupling  -format SPEF  $SPEF_FILE

source -e -v $SDC_FILE

set_propagated_clock [all_clocks]

set timing_remove_clock_reconvergence_pessimism true

set timing_disable_clock_gating_checks true  
set timing_report_unconstrained_paths true

update_timing -full

report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 100000 -slack_lesser_than 10 > ../PtRpt/${top_design}.rpt

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]
exit
