import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import time
import threading
import pickle
import numpy as np
import Global_var
import PtRpt_Parser
import TimingLibTrans
import ItfTrans

class Optimize_Data_Prepare:
    def __init__(self):
        self.FEOL_Corners = Global_var.FEOL_Corners
        self.BEOL_Corners = Global_var.BEOL_Corners
        self.LibTrans = TimingLibTrans.TimingLibTrans()
        self.ItfTrans = ItfTrans.ItfTrans()
        self.base_corner = Global_var.base_corner
        
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
    
    def Prepare_Data(self, design):
        inrpt = os.path.join(Global_var.PtRpt_Path, f"{design}.rpt")
        paths = PtRpt_Parser.Read_PtRpt(inrpt)
        
        Itf_data = self.ItfTrans.load_normalized_data()
        Lib_data = self.LibTrans.load_normalized_data()
        
        all_done = threading.Event()
        status_thread = threading.Thread(target=self._output_status, args=(all_done, "Processing base path and target path data"))
        status_thread.start()
        
        try:
            dataset = []
            
            base_lib_key = (self.base_corner[0], self.base_corner[1], self.base_corner[2])    
            base_lib_data = Lib_data[base_lib_key]
            
            for path in paths:
                fpath = [path.pathdelay, path.slack, path.required_time, path.clk.delay_start, path.clk.delay_end, path.clk.T]
                if path.constrain == None:
                    fpath.append(0)
                else:
                    fpath.append(path.setup)
                base_delay, base_trans = [], []
                for cellarc in path.Cellarcs:
                    cell = cellarc.cell
                    parts = cellarc.name.split("->")
                    part1 = parts[0].split('/')[1]
                    part2 = parts[1].split('/')[1]
                    cellarc_key = f"{part1}->{part2}"
                    rf = cellarc.rf
                    base_delay.append([base_lib_data[cell].delay[(cellarc_key, rf)].index1, base_lib_data[cell].delay[(cellarc_key, rf)].index2, base_lib_data[cell].delay[(cellarc_key, rf)].delay])
                    base_trans.append([base_lib_data[cell].trans[(cellarc_key, rf)].index1, base_lib_data[cell].trans[(cellarc_key, rf)].index2, base_lib_data[cell].trans[(cellarc_key, rf)].trans])
                
                for FEOL in self.FEOL_Corners:
                    for BEOL in self.BEOL_Corners:
                        if FEOL + (BEOL,) == self.base_corner:
                            continue
                        
                        # fpath = [path.pathdelay, path.slack, path.required_time, path.clk.delay_start, path.clk.delay_end, path.clk.T]
                        # if path.constrain == None:
                        #     fpath.append(0)
                        # else:
                        #     fpath.append(path.setup)
                        
                        target_lib_key = (FEOL[0], FEOL[1], FEOL[2])
                        target_lib_data = Lib_data[target_lib_key]
                        # base_delay, base_trans = [], []
                        target_delay, target_trans = [], []
                        for cellarc in path.Cellarcs:
                            cell = cellarc.cell
                            parts = cellarc.name.split("->")
                            part1 = parts[0].split('/')[1]
                            part2 = parts[1].split('/')[1]
                            cellarc_key = f"{part1}->{part2}"
                            rf = cellarc.rf
                            # base_delay.append([base_lib_data[cell].delay[(cellarc_key, rf)].index1, base_lib_data[cell].delay[(cellarc_key, rf)].index2, base_lib_data[cell].delay[(cellarc_key, rf)].delay])
                            # base_trans.append([base_lib_data[cell].trans[(cellarc_key, rf)].index1, base_lib_data[cell].trans[(cellarc_key, rf)].index2, base_lib_data[cell].trans[(cellarc_key, rf)].trans])
                            target_delay.append([target_lib_data[cell].delay[(cellarc_key, rf)].index1, target_lib_data[cell].delay[(cellarc_key, rf)].index2, target_lib_data[cell].delay[(cellarc_key, rf)].delay])
                            target_trans.append([target_lib_data[cell].trans[(cellarc_key, rf)].index1, target_lib_data[cell].trans[(cellarc_key, rf)].index2, target_lib_data[cell].trans[(cellarc_key, rf)].trans])
                        if path.constrain == None:
                            base_constrain = np.zeros((3, 3, 3)).tolist()
                            target_constrain = np.zeros((3, 3, 3)).tolist()
                        else:
                            constrain = base_lib_data[path.constrain.cell].constrain[('setup', path.constrain.rf)]
                            base_constrain = [constrain.index1, constrain.index2, constrain.constrain]
                            constrain = target_lib_data[path.constrain.cell].constrain[('setup', path.constrain.rf)]
                            target_constrain = [constrain.index1, constrain.index2, constrain.constrain]
                        
                        data = [FEOL + (BEOL, path.Endpoint), base_delay, base_trans, base_constrain, target_delay, target_trans, target_constrain, fpath, Itf_data[self.base_corner[3]], Itf_data[BEOL], [0, 0]]
                        dataset.append(data)
        except Exception as e:
            print(f"Error occurred: {e}")
            all_done.set()
            raise
        else:
            all_done.set()
            self._save_dataset(design, dataset)
        
    def _save_dataset(self, design, dataset):
        Save_Path = Global_var.Dataset_Path
        save_dir = os.path.join(Save_Path, f"Optimization_Dataset/{self.base_corner[0]}_{self.base_corner[1]}_{self.base_corner[2]}_{self.base_corner[3]}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{design}_data")
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Optimization dataset of {design} prepared.")
    
    def load_dataset(self, design):
        print(f"Loading optimization dataset of {design}.")
        Save_Path = Global_var.Dataset_Path
        save_dir = os.path.join(Save_Path, f"Optimization_Dataset/{self.base_corner[0]}_{self.base_corner[1]}_{self.base_corner[2]}_{self.base_corner[3]}")
        save_path = os.path.join(save_dir, f"{design}_data")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Optimization data of {self.design} not found, using Prepare_Data to rebuild data.")
        with open(save_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Optimization dataset of {design} loaded.")
        return dataset
    
if __name__ == "__main__":
    data_preparer = Optimize_Data_Prepare()
    data_preparer.Prepare_Data("aes_cipher_top")
    dataset = data_preparer.load_dataset("aes_cipher_top")
    # print(len(dataset))
    # for data in dataset:
    #     print(data[1], data[4])