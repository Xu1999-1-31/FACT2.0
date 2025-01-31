set starttime [clock seconds]                                                                                                            
echo "INFORM: Start job at: " [clock format $starttime -gmt false]

set bench aes_cipher_top
#set bench spi_top
open_lib ../Icc2Ndm/${bench}_nlib
open_block ${bench}_eco
link_block

set fill_cell_list "*/FILL*"
if {[file exists ../PtECO/${bench}/change.tcl]} {
    source ../PtECO/${bench}/change.tcl
}

place_eco_cells -eco_changed_cells -legalize_only -legalize_mode minimum_physical_impact -remove_filler_references $fill_cell_list
set_app_options -name route.global.timing_driven    -value false
set_app_options -name route.track.timing_driven     -value false
set_app_options -name route.detail.timing_driven    -value false 
set_app_options -name route.global.crosstalk_driven -value false                                                                                  
set_app_options -name route.track.crosstalk_driven  -value false
route_eco -utilize_dangling_wires true -reroute modified_nets_first_then_others -open_net_driven true
create_stdcell_fillers -lib_cells $fill_cell_list
connect_pg_net
save_block
exit