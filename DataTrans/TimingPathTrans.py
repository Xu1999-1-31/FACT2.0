import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import pickle
import threading
import time
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import Global_var
import PtRpt_Parser

class Path():
    def __init__(self):
        self.fpath = None
        self.Cellarcs = []
        self.Startpoint = ""
        self.Endpoint = ""
        self.constrain = ""
        self.slack = 0
        self.delay = 0

class TimingPathTrans():
    def __init__(self, design):
        self.FEOL_Corners = Global_var.FEOL_Corners
        self.BEOL_Corners = Global_var.BEOL_Corners
        self.design = design
    
    def _output_status(self, all_done, status):
        dot_count = 0
        while not all_done.is_set():
            dot_count = dot_count + 1 if dot_count < 6 else 0
            if(dot_count == 0):
                print(f"\r{status} " + " " * 6, end="")
            else:
                print(f"\r{status} " + "." * dot_count, end="")
                time.sleep(1.5)
        print()
    
    def _read_paths(self, FEOL, BEOL):
        PtRpt_Path = os.path.join(Global_var.PtRpt_Path, f"{self.design}/func_{FEOL[0]}_{FEOL[1]}_{FEOL[2]}_{BEOL}")
        inrpt = os.path.join(PtRpt_Path, "timing.rpt")
        paths = PtRpt_Parser.Read_PtRpt(inrpt)
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, f"TimingPath/{self.design}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{FEOL[0]}{FEOL[1]}{FEOL[2]}_{BEOL}")
        with open(save_path, "wb") as f:
            pickle.dump(paths, f)
            
    def _read_saved_paths(self):
        print(f"Reading Saved Timing Paths Data.")
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, f"TimingPath/{self.design}")
        path_data = {}
        for FEOL in self.FEOL_Corners:
            for BEOL in self.BEOL_Corners:
                save_path = os.path.join(save_dir, f"{FEOL[0]}{FEOL[1]}{FEOL[2]}_{BEOL}")
                if not os.path.exists(save_path):
                    raise FileNotFoundError(f"Timing path data of {FEOL[0]}{FEOL[1]}{FEOL[2]}_{BEOL} not found, using reRead to update timing paths.")
                with open(save_path, "rb") as f:
                    path_data[(FEOL[0], FEOL[1], FEOL[2], BEOL)] = pickle.load(f)
        
        return path_data
    
    """"""""""""""""""""""""""""""""""""
    """ Main function to process paths"""
    """"""""""""""""""""""""""""""""""""   
    def path_processer(self, reRead = True):
        if reRead:
            all_done = threading.Event()
            status_thread = threading.Thread(target=self._output_status, args=(all_done, "Reading Pt Rpts"))
            status_thread.start()

            try:
                max_processes = 8
                processes = []
                for FEOL in self.FEOL_Corners:
                    for BEOL in self.BEOL_Corners:
                        if len(processes) >= max_processes:
                            processes[0].join()
                            processes.pop(0)
                        process = Process(target=self._read_paths, args=(FEOL, BEOL))
                        processes.append(process)
                        process.start()
                for process in processes:
                    process.join()
                    
                all_done.set()
            except Exception as e:
                print(f"Error occurred: {e}")
                all_done.set()
                raise
            status_thread.join()

        else:
            print(f"Using Saved Timing Paths.")
        
        # normalization
        path_data = self._read_saved_paths()
        Fpath = [] # path delay, path slack, clock network delay, clock cycle, constrain time, temperature
        for key, paths in path_data.items():
            for path in paths:
                fpath = [path.pathdelay, path.slack, path.required_time, path.clk.delay_start, path.clk.delay_end, path.clk.T]
                if path.constrain == None:
                    fpath.append(0)
                else:
                    fpath.append(path.setup)
                # if 'm' in key[2]:
                #     fpath.append(-40)
                # else:
                #     fpath.append(float(key[2]))
                Fpath.append(fpath)
        # scaler = MinMaxScaler()
        # Fpath = scaler.fit_transform(np.array(Fpath)).tolist()
        
        count = 0
        newpath_data = {}
        for key, paths in path_data.items():
            newpaths = []
            for path in paths:
                newpath = Path()
                newpath.fpath = Fpath[count]
                newpath.Cellarcs = path.Cellarcs
                newpath.Startpoint = path.Startpoint
                newpath.Endpoint = path.Endpoint
                newpath.constrain = path.constrain
                newpath.slack = path.slack
                newpath.delay = path.pathdelay
                newpaths.append(newpath)
                count += 1
            newpath_data[key] = newpaths
        
        self._save_pathdata_with_fpath(newpath_data)
        
    def _save_pathdata_with_fpath(self, path_data):
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingPath_Normalized")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.design}_path")
        with open(save_path, "wb") as f:
            pickle.dump(path_data, f)
    
    def load_normalized_data(self):
        print(f"Loading timing path data of {self.design}.")
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingPath_Normalized")
        save_path = os.path.join(save_dir, f"{self.design}_path")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Path data of {self.design} not found, using path_processer to rebuild data.")
        with open(save_path, "rb") as f:
            path_data = pickle.load(f)
        return path_data
        
if __name__ == "__main__":
    # PathProcesser = TimingPathTrans("aes_cipher_top")
    # PathProcesser.path_processer(reRead=True)
    for design in Global_var.Designs:
        PathProcesser = TimingPathTrans(design)
        path_data = PathProcesser._read_saved_paths()
        max_length = 0
        for corner, paths in path_data.items():
            for path in paths:
                if len(path.Cellarcs) > max_length:
                    max_length = len(path.Cellarcs)
        print(f"MLP for {design}: {max_length}")