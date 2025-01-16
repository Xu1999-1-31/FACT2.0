import argparse
import Global_var
import tee
import TrainDataPrepare
import TimingArc_Encoding
import MmmcPredictor_Training
import Interaction


parser = argparse.ArgumentParser(description='This is the main python program for FACT project.')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--reload', nargs='+', type=str, help='Specify one or more designs to reload data')
parser.add_argument('--reload_all', action='store_true', help='reload all design data')
parser.add_argument('--rebuild_tech', action='store_true', help='Rebuild all technical files')
parser.add_argument('--stdout', type=str, help='File path to redirect stdout')
parser.add_argument('--stderr', type=str, help='File path to redirect stderr')
parser.add_argument('--disable-log', action='store_true', help='Disable logging to file')
parser.add_argument('--optimize', type=str, help='Specify a design to optimize')


args = parser.parse_args()

loggers = [] # logger to store iostream

if not args.disable_log:
    # set stdout
    if args.stdout:
        stdout_file = args.stdout
    else:
        stdout_file = 'stdout.log'  # 默认文件名
    
    # redirect stdout
    stdout_logger = tee.StdoutTee(stdout_file, mode="w")
    loggers.append(stdout_logger)

    # set stderr
    if args.stderr:
        stderr_file = args.stderr
    else:
        stderr_file = 'stderr.log'  # 默认文件名

    # redirect stderr
    stderr_logger = tee.StderrTee(stderr_file, mode="w")
    loggers.append(stderr_logger)

# main program
if __name__ == '__main__':
    TrainDataProcesser = TrainDataPrepare.Train_Data_Prepare()
    if args.rebuild_tech:
        TrainDataProcesser.RebuildItf()
        TrainDataProcesser.ReBuildLib()
    if args.reload_all:
        for design in Global_var.Designs:
            TrainDataProcesser.Prepare_Data(design, rebuild=False)
            TimingArc_Encoding.TimingArc_Encoder(design)
    elif args.reload:
        for design in args.reload:
            TrainDataProcesser.Prepare_Data(design, rebuild=False)
            TimingArc_Encoding.TimingArc_Encoder(design)
        
    if args.train:
        with tee.StdoutTee(stdout_file), tee.StderrTee(stderr_file):
            MmmcPredictor_Training.TrainPredictor()
    if args.test:
        with tee.StdoutTee(stdout_file), tee.StderrTee(stderr_file):
            MmmcPredictor_Training.TestModel()
    if args.optimize:
        Interaction.Run_Pt_Script(f"{design}_pt_rpt.tcl", MMMC=False)
    
    

# close all loggers
tee.close_all()
