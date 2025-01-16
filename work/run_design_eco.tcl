set link_library "* Timing_Lib/tcbn28hpcplusbwp7t40p140${corner}${voltage}${temp}c_ccs.db"

read_verilog Icc2Output/${top_design}_eco.v
current_design $top_design
link

set_operating_conditions -analysis_type on_chip_variation ${corner}${voltage}${temp}c

switch $rc_corner {
rcbest {
 read_parasitics -keep_capacitive_coupling -format SPEF Icc2Output/${top_design}_eco.rcbest_-40.spef
}
rcworst {
 read_parasitics -keep_capacitive_coupling -format SPEF Icc2Output/${top_design}_eco.rcworst_125.spef
}
cbest {
 read_parasitics -keep_capacitive_coupling -format SPEF Icc2Output/${top_design}_eco.cbest_-40.spef
}
cworst {
 read_parasitics -keep_capacitive_coupling -format SPEF Icc2Output/${top_design}_eco.cworst_125.spef
} 
typical {
 read_parasitics -keep_capacitive_coupling -format SPEF Icc2Output/${top_design}_eco.typical_25.spef
}
}

switch $mode {
func {
    source -e -v ../../../Icc2Output/${top_design}_eco.sdc
    set is_si_enabled false
    source -e -v ../../../pt_variable.tcl
    set_propagated_clock [all_clocks]
    set timing_remove_clock_reconvergence_pessimism true
    set timing_disable_clock_gating_checks true  
    set timing_report_unconstrained_paths true
}
}

set_eco_options -physical_icc2_lib /home/jiajiexu/Project/FACT/work/Icc2Ndm/${top_design}_nlib -physical_icc2_blocks ${top_design}_eco.design
set_app_var eco_allow_filler_cells_as_open_sites true                                                                                             
set_app_var eco_enable_fixing_clock_used_as_data true
set_eco_options -physical_enable_clock_data