import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import numpy as np
import time
import threading
import pickle
import Global_var
import TimingLibTrans
import TimingPathTrans
import ItfTrans
from collections import defaultdict

class Train_Data_Prepare:
    def __init__(self):
        self.FEOL_Corners = Global_var.FEOL_Corners
        self.BEOL_Corners = Global_var.BEOL_Corners
        self.LibTrans = TimingLibTrans.TimingLibTrans()
        self.ItfTrans = ItfTrans.ItfTrans()
        self.base_corner = Global_var.base_corner
    
    def RebuildItf(self):
        self.ItfTrans.itf_processer(reRead=True)
    
    def ReBuildLib(self):
        self.LibTrans.lib_processer(reRead=True)
    
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
    
    def Prepare_Data(self, design, rebuild = False):
        PathTrans = TimingPathTrans.TimingPathTrans(design)
        if rebuild:
            PathTrans.path_processer(reRead=True)
        
        all_done = threading.Event()
        status_thread = threading.Thread(target=self._output_status, args=(all_done, "Reading normalized path data"))
        status_thread.start()
        try:
            path_data = PathTrans.load_normalized_data()
        except Exception as e:
            print(f"Error occurred: {e}")
            all_done.set()
            raise    
        else:
            all_done.set()
            Itf_data = self.ItfTrans.load_normalized_data()
            Lib_data = self.LibTrans.load_normalized_data()
            base_corner_paths = path_data[self.base_corner]
            dataset = [] # training dataset for the design
            # corners, fpath, base_itf, target_itf = [], [], [], []
            # base_delays, base_transes, base_constrains, target_delays, target_transes, target_constrains = [], [], [], [], [], []
            
            all_done = threading.Event()
            status_thread = threading.Thread(target=self._output_status, args=(all_done, "Processing training data"))
            status_thread.start()
            
            try:
                base_lib_key = (self.base_corner[0], self.base_corner[1], self.base_corner[2])
                base_lib_data = Lib_data[base_lib_key]
                
                path_map = defaultdict(list)
                
                for corner, paths in path_data.items():
                    for path in paths:
                        cellarc_keys = tuple(
                            (
                                cellarc.cell,
                                "->".join([part.split('/')[1] for part in cellarc.name.split("->")]),
                                cellarc.rf
                            ) 
                            for cellarc in path.Cellarcs
                        )
                        key = (cellarc_keys, path.Startpoint, path.Endpoint)
                        path_map[key].append((corner, path))
                        
                for base_path in base_corner_paths:
                    base_cellarc_keys = tuple(
                        (
                            cellarc.cell,
                            "->".join([part.split('/')[1] for part in cellarc.name.split("->")]),
                            cellarc.rf
                        ) 
                        for cellarc in base_path.Cellarcs
                    )
                    key = (base_cellarc_keys, base_path.Startpoint, base_path.Endpoint)
                    matching_paths = path_map.get(key, [])
                    # count = 0
                    for corner, path in matching_paths:
                        if corner == self.base_corner:
                            continue
                        target_lib_key = (corner[0], corner[1], corner[2])
                        target_lib_data = Lib_data[target_lib_key]
                        base_delay, target_delay, base_trans, target_trans = [], [], [], []
                        for cellarc in base_path.Cellarcs:
                            cell = cellarc.cell
                            parts = cellarc.name.split("->")
                            part1 = parts[0].split('/')[1]
                            part2 = parts[1].split('/')[1]
                            cellarc_key = f"{part1}->{part2}"
                            rf = cellarc.rf

                            base_delay.append([base_lib_data[cell].delay[(cellarc_key, rf)].index1, base_lib_data[cell].delay[(cellarc_key, rf)].index2, base_lib_data[cell].delay[(cellarc_key, rf)].delay])
                            base_trans.append([base_lib_data[cell].trans[(cellarc_key, rf)].index1, base_lib_data[cell].trans[(cellarc_key, rf)].index2, base_lib_data[cell].trans[(cellarc_key, rf)].trans])
                            target_delay.append([target_lib_data[cell].delay[(cellarc_key, rf)].index1, target_lib_data[cell].delay[(cellarc_key, rf)].index2, target_lib_data[cell].delay[(cellarc_key, rf)].delay])
                            target_trans.append([target_lib_data[cell].trans[(cellarc_key, rf)].index1, target_lib_data[cell].trans[(cellarc_key, rf)].index2, target_lib_data[cell].trans[(cellarc_key, rf)].trans])
                        if base_path.constrain == None:
                            base_constrain = np.zeros((3, 3, 3)).tolist()
                            target_constrain = np.zeros((3, 3, 3)).tolist()
                        else:
                            constrain = base_lib_data[base_path.constrain.cell].constrain[('setup', base_path.constrain.rf)]
                            base_constrain = [constrain.index1, constrain.index2, constrain.constrain]
                            constrain = target_lib_data[path.constrain.cell].constrain[('setup', path.constrain.rf)]
                            target_constrain = [constrain.index1, constrain.index2, constrain.constrain]
                        # the data needed is [base delay_seq, base trans seq, base constrain, target delay seq, target trans seq, target constrain, fpath, itf1, itf2, slack and delay]
                        data = [corner, base_delay, base_trans, base_constrain, target_delay, target_trans, target_constrain, base_path.fpath, Itf_data[self.base_corner[3]], Itf_data[corner[3]], [path.slack, path.delay]]
                        dataset.append(data)
                    #     count += 1
                    # print(count)

            except Exception as e:
                print(f"Error occurred: {e}")
                all_done.set()
                status_thread.join()
                raise    
            else:
                all_done.set()
                status_thread.join()
                self._save_dataset(design, dataset)
        
    def _save_dataset(self, design, dataset):
        Save_Path = Global_var.Dataset_Path
        save_dir = os.path.join(Save_Path, f"Processed_Dataset/{self.base_corner[0]}_{self.base_corner[1]}_{self.base_corner[2]}_{self.base_corner[3]}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{design}_data")
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Dataset of {design} prepared.")
    
    def load_dataset(self, design):
        print(f"Loading training dataset of {design}.")
        Save_Path = Global_var.Dataset_Path
        save_dir = os.path.join(Save_Path, f"Processed_Dataset/{self.base_corner[0]}_{self.base_corner[1]}_{self.base_corner[2]}_{self.base_corner[3]}")
        save_path = os.path.join(save_dir, f"{design}_data")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Training data of {design} not found, using Prepare_Data to rebuild data.")
        with open(save_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Dataset of {design} loaded.")
        return dataset
        
if __name__ == "__main__":
    data_preparer = Train_Data_Prepare()
    for design in Global_var.Designs:
        data_preparer.Prepare_Data(design, rebuild=True)
    # data_preparer.Prepare_Data("tate_pairing", rebuild=True)
    # dataset = data_preparer.load_dataset("aes_cipher_top")
    # print(len(dataset))
