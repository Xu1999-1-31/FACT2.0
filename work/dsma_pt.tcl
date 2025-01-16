# xujj
set starttime [clock seconds]                                                                               
echo "INFORM: Start job at: " [clock format $starttime -gmt false]
set_app_var eco_enable_more_scenarios_than_hosts true

# set the working directory and error files (delete the old work directory first)
set top_design aes_cipher_top
file delete -force ../PtRpt/${top_design}
set multi_scenario_working_directory ../PtRpt/${top_design}
set multi_scenario_merged_error_log ../PtRpt/${top_design}/error_log.txt

# add search path for design scripts (scenarios will
# inherit the master's search_path)
lappend search_path ..


# add slave workstation information
#
# NOTE: change this to your desired machine/add more machines!
# -farm is a required option, valid types are:
#
# -farm now (network of discrete workstations)
set_host_options -load_factor 2 -num_processes 16 [info hostname] -max_cores 1

set pairs {}
foreach corner {ffg} {
  foreach voltage {0p88v 0p99v 1p05v} {
    foreach temp {125 0 m40} {
      lappend pairs [list $corner $voltage $temp]
    }
  }
}

foreach corner {ssg} {
  foreach voltage {0p72v 0p81v 0p9v} {
    foreach temp {125 0 m40} {
      lappend pairs [list $corner $voltage $temp]
    }
  }
}

foreach corner {tt} {
  foreach voltage {0p8v 0p9v 1v} {
    foreach temp {25 85} {
      lappend pairs [list $corner $voltage $temp]
    }
  }
}

foreach pair $pairs {
 foreach mode {func} {
  foreach rc_corner {rcworst rcbest cworst cbest typical} {
   set corner [lindex $pair 0]
   set voltage [lindex $pair 1]
   set temp [lindex $pair 2]

   create_scenario \
    -name ${mode}_${corner}_${voltage}_${temp}_${rc_corner} \
    -specific_variables {top_design mode corner voltage temp rc_corner} \
    -specific_data {run_design.tcl} \
  }
 }
}

# start processes on all remote machines
#
# if this hangs, check to make sure that you can run this version
# of PrimeTime on the specified machines/farm
start_hosts

# set session focus to all scenarios
current_session -all

# send some non-merged reports to our slave processes
remote_execute {
 update_timing -full
 #source ../../../PtScripts/${top_design}_Rpt.tcl > timing.rpt
 #report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -pba_mode exhaustive -max_paths 1000000 -nworst 5 -slack_lesser_than 10 > timing.rpt
 report_timing -nosplit -nets -input_pins -transition_time -capacitance -significant_digit 6 -max_path 1000 -pba_mode exhaustive -nworst 5 -slack_lesser_than 10 > timing.rpt
}

# now execute a merged report
#report_timing -nets -input_pins -transition_time -capacitance -significant_digit 9 -max_path 1000 -crosstalk_delta > timing_max.rpt
#report_timing -nets -input_pins -transition_time -capacitance -significant_digit 9 -max_path 1000 -crosstalk_delta -delay_type min > timing_min.rpt

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]
exit 
