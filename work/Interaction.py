import subprocess
import shutil
import signal
import os
import atexit
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import OptimizeDataPrepare
import TimingArc_Encoding
import MmmcPredictor_Training

def check_command(command_name):
    # check if the command is available
    return shutil.which(command_name) is not None

def Run_Pt_Script(script, MMMC=False):
    """
    Run a PrimeTime (pt_shell) script using subprocess, ensuring child processes
    are terminated when the main process exits or is killed.

    Args:
        script (str): The name of the script to run.
    """
    if not check_command("pt_shell"):
        raise EnvironmentError("Error: 'pt_shell' not found. Please ensure PrimeTime is correctly installed and added to your PATH.")
    if MMMC:
        command = ["pt_shell", "-multi_scenario -f", "../" + script]
    else:
        command = ["pt_shell", "-f", "../" + script]
    working_directory = os.path.join(Global_var.work_dir, "log/")
    
    # Create the working directory if it doesn't exist
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    print(f"Running PT script: {script} in {working_directory}")
    
    # Start the subprocess and assign it to a new process group
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Register cleanup to terminate child processes on exit
    def cleanup():
        if process.poll() is None:  # Check if the process is still running
            print("Terminating PT subprocess...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
            print("PT subprocess terminated.")
    
    atexit.register(cleanup)  # Ensure cleanup is called on normal program exit

    # Handle signals such as Ctrl+C
    def handle_signal(signum, frame):
        print(f"Received signal {signal.Signals(signum).name}, cleaning up...")
        cleanup()
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Handle kill signals

    try:
        # Read and print the subprocess output and error streams
        for line in process.stdout:
            print(line, end="")  # Print standard output
        for err_line in process.stderr:
            print(err_line, end="")  # Print standard error
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating PT script...")
        cleanup()
    finally:
        process.wait()  # Wait for the process to complete
    
    # Check the subprocess return code
    print("Return code:", process.returncode)


def Run_Icc2_Script(script):
    """
    Run an IC Compiler II (icc2_shell) script using subprocess, ensuring child processes
    are terminated when the main process exits or is killed.

    Args:
        script (str): The name of the script to run.
    """
    if not check_command("icc2_shell"):
        raise EnvironmentError("Error: 'icc2_shell' not found. Please ensure IC Compiler II is correctly installed and added to your PATH.")
    
    command = ["icc2_shell", "-f", "../" + script]
    working_directory = os.path.join(Global_var.work_dir, "log/")
    
    # Create the working directory if it doesn't exist
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    print(f"Running ICC2 script: {script} in {working_directory}")
    
    # Start the subprocess and assign it to a new process group
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Register cleanup to terminate child processes on exit
    def cleanup():
        if process.poll() is None:  # Check if the process is still running
            print("Terminating ICC2 subprocess...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
            print("ICC2 subprocess terminated.")
    
    atexit.register(cleanup)  # Ensure cleanup is called on normal program exit

    # Handle signals such as Ctrl+C
    def handle_signal(signum, frame):
        print(f"Received signal {signal.Signals(signum).name}, cleaning up...")
        cleanup()
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Handle kill signals

    try:
        # Read and print the subprocess output and error streams
        for line in process.stdout:
            print(line, end="")  # Print standard output
        for err_line in process.stderr:
            print(err_line, end="")  # Print standard error
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating ICC2 script...")
        cleanup()
    finally:
        process.wait()  # Wait for the process to complete
    
    # Check the subprocess return code
    print("Return code:", process.returncode)

def Write_Icc2_Scripts(design):
    # read initial file
    print(f"Generated ICC2 script for {design}")
    filenames = ["icc2_rpt.tcl", "copy_block.tcl", "icc2_eco_rpt.tcl", "implement_eco.tcl"] # scripts for writing icc2 verilog, copy initial block to eco block, writing verilog after eco, implement eco changes
    for filename in filenames:
        file = os.path.join(Global_var.work_dir, filename)
        with open(file, "r") as file:
            script_content = file.read()

        # replace initial design to new design
        updated_content = script_content.replace("set bench aes_cipher_top", f"set bench {design}")

        # write new file
        script_filename = os.path.join(Global_var.work_dir, f"{design}_{filename}")
        with open(script_filename, "w") as new_file:
            new_file.write(updated_content)

def Write_Pt_Scripts(design):
    # read initial file
    print(f"Generated PT script for {design}")
    file = os.path.join(Global_var.work_dir, "pt_rpt.tcl")
    with open(file, "r") as file:
        script_content = file.read()

    # replace initial design to new design
    updated_content = script_content.replace("set top_design aes_cipher_top", f"set top_design {design}")

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f"{design}_pt_rpt.tcl")
    with open(script_filename, "w") as new_file:
        new_file.write(updated_content)

def Write_Pt_MMMC_Scripts(design):
    # read initial file
    print(f"Generated MMMC PT script for {design}")
    filenames = ["dsma_pt.tcl", "dsma_pt_eco.tcl", "dsma_pt_global.tcl"]
    for filename in filenames:
        file = os.path.join(Global_var.work_dir, filename)
        with open(file, "r") as file:
            script_content = file.read()

        # replace initial design to new design
        updated_content = script_content.replace("set top_design aes_cipher_top", f"set top_design {design}")

        # write new file
        script_filename = os.path.join(Global_var.work_dir, f"{design}_{filename}")
        with open(script_filename, "w") as new_file:
            new_file.write(updated_content)

def Delete_Temp_Scripts(design):
    filenames = ["icc2_rpt.tcl", "copy_block.tcl", "icc2_eco_rpt.tcl", "implement_eco.tcl"]
    for filename in filenames:
        path = os.path.join(Global_var.work_dir, f"{design}_{filename}")
        if os.path.exists(path):
            os.remove(path)
    path = os.path.join(Global_var.work_dir, f"{design}_pt_rpt.tcl")
    if os.path.exists(path):
        os.remove(path)
    filenames = ["dsma_pt.tcl", "dsma_pt_eco.tcl", "dsma_pt_global.tcl"]
    for filename in filenames:
        path = os.path.join(Global_var.work_dir, f"{design}_{filename}")
        if os.path.exists(path):
            os.remove(path)
        
    print(f"Deleted temporary scripts for {design}")
        

def Build_MMMC_Data(design):
    Write_Icc2_Scripts(design)
    Run_Icc2_Script(f"{design}_icc2_rpt.tcl")
    # Write_Pt_Scripts(design)
    # Run_Pt_Script(f"{design}_pt_rpt.tcl", MMMC=False)
    # BuildPtScripts.BuildPtScripts(design)
    Write_Pt_MMMC_Scripts(design)
    Run_Pt_Script(f"{design}_mmmc_pt_rpt.tcl", MMMC=True)
    Delete_Temp_Scripts(design)

def Optimize_Design(design, incremental=False):
    Write_Icc2_Scripts(design)
    Write_Pt_MMMC_Scripts(design)
    if not incremental: # new eco iteration
        Run_Icc2_Script(f"{design}_copy_block.tcl")   
    
    # # write timing report for corner selection
    # Write_Pt_Scripts(design)
    # Run_Pt_Script(f"{design}_pt_rpt.tcl", MMMC=False)
    
    # # predict MMMC timing
    # data_preparer = OptimizeDataPrepare.Optimize_Data_Prepare()
    # data_preparer.Prepare_Data(design)
    # TimingArc_Encoding.TimingArc_Encoder(design, optimize=True)
    # Predictor = MmmcPredictor_Training.loadPredictor()
    # # ... predict timing and choose the critical corners
    
    # call pt for optimization
    Run_Pt_Script(f"{design}_dsma_pt_eco.tcl", MMMC=True)
    
    # # implement eco changes
    # Run_Icc2_Script(f"{design}_implement_eco.tcl")
    
    # # write eco verilog and sdc
    # Run_Icc2_Script(f"{design}_icc2_eco_rpt.tcl") 
    
    # # report tns after eco
    # Run_Pt_Script(f"{design}_dsma_pt_global.tcl", MMMC=True)
    
    Delete_Temp_Scripts(design)
    
def Report_Initial_TNS(design):
    Write_Icc2_Scripts(design)
    Write_Pt_MMMC_Scripts(design)
    Run_Icc2_Script(f"{design}_copy_block.tcl")
    Run_Icc2_Script(f"{design}_icc2_eco_rpt.tcl")
    Run_Pt_Script(f"{design}_dsma_pt_global.tcl", MMMC=True)
    Delete_Temp_Scripts(design)
    
    
    
    

    
    
if __name__ == "__main__": 
    # Build_MMMC_Data("sasc_top")
    # Report_Initial_TNS("spi_top")
    Optimize_Design("spi_top", incremental=True)
