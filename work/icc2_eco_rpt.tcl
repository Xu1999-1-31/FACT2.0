set starttime [clock seconds]                                                                                                            
echo "INFORM: Start job at: " [clock format $starttime -gmt false]


set bench aes_cipher_top
#set bench spi_top
open_lib ../Icc2Ndm/${bench}_nlib
open_block ${bench}_eco
link_block

# report_net -physical -significant_digits 6 -verbose > ../Icc2Rpt/${bench}_net.rpt

# set bboxs [get_attribute -objects [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"] -name boundary]

# set cells [get_object_name [get_cell -filter "(ref_name !~ *FILL*) && (ref_name !~ *ENDCAP*) && (ref_name !~ *DCAP*)"]]

# if {[file exists ../Icc2Rpt/${bench}_cell.rpt]} {
#     file delete -force ../Icc2Rpt/${bench}_cell.rpt
# }                                                                                                                                        

# #set file_output [open ./outputs_rpt/${bench}_cell.rpt w]

# foreach cell $cells {
#     echo $cell >> ../Icc2Rpt/${bench}_cell.rpt
# }


# foreach bbox $bboxs {
#     echo $bbox >> ../Icc2Rpt/${bench}_cell.rpt
# }

# source ../list_pin_bbox.tcl > ../Icc2Rpt/${bench}_pin.rpt
# source ../list_port_bbox.tcl > ../Icc2Rpt/${bench}_port.rpt

# source ../list_drc_errors.tcl > ../Icc2Rpt/${bench}_drc.rpt

# set dimens 128
# source ../report_congestion.tcl > ../Icc2Rpt/${bench}_congestion_${dimens}.rpt

#read_parasitic_tech -name {rcworst} -tlup {/home/md1/eda/techlibs/TSMC/28nm/TF/RC_TLUplus_cln28hpc+1p9m/cln28hpc+_1p09m+ut-alrdl_4x2y2r_rcworst.tluplus} -layermap {/home/md1/eda/techlibs/TSMC/28nm/TF/PDK/CCI/online/1P9M_4X2Y2R/starrcxt_mapping}
#read_parasitic_tech -name {rcbest} -tlup {/home/md1/eda/techlibs/TSMC/28nm/TF/RC_TLUplus_cln28hpc+1p9m/cln28hpc+_1p09m+ut-alrdl_4x2y2r_rcbest.tluplus} -layermap {/home/md1/eda/techlibs/TSMC/28nm/TF/PDK/CCI/online/1P9M_4X2Y2R/starrcxt_mapping}
#read_parasitic_tech -name {cworst} -tlup {/home/md1/eda/techlibs/TSMC/28nm/TF/RC_TLUplus_cln28hpc+1p9m/cln28hpc+_1p09m+ut-alrdl_4x2y2r_cworst.tluplus} -layermap {/home/md1/eda/techlibs/TSMC/28nm/TF/PDK/CCI/online/1P9M_4X2Y2R/starrcxt_mapping}
#read_parasitic_tech -name {cbest} -tlup {/home/md1/eda/techlibs/TSMC/28nm/TF/RC_TLUplus_cln28hpc+1p9m/cln28hpc+_1p09m+ut-alrdl_4x2y2r_cbest.tluplus} -layermap {/home/md1/eda/techlibs/TSMC/28nm/TF/PDK/CCI/online/1P9M_4X2Y2R/starrcxt_mapping}
#read_parasitic_tech -name {typical} -tlup {/home/md1/eda/techlibs/TSMC/28nm/TF/RC_TLUplus_cln28hpc+1p9m/cln28hpc+_1p09m+ut-alrdl_4x2y2r_typical.tluplus} -layermap {/home/md1/eda/techlibs/TSMC/28nm/TF/PDK/CCI/online/1P9M_4X2Y2R/starrcxt_mapping}  


write_verilog -exclude {cover_cells well_tap_cells filler_cells end_cap_cells corner_cells } ../Icc2Output/${bench}_eco.v
write_sdc -output ../Icc2Output/${bench}_eco.sdc
write_def  ../Icc2Output/${bench}_eco.def
write_lef  -include {cell tech} ../Icc2Output/${bench}_eco.lef

#set_parasitics_parameters -early_spec rcbest -late_spec rcworst
#update_timing
set_app_option -name extract.enable_coupling_cap -value true
write_parasitics  -output ../Icc2Output/${bench}_eco 

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]


exit
