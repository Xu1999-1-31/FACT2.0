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
import TimingLib_Parser

class TimingLibTrans():
    def __init__(self):
        self.FEOL_Corners = Global_var.FEOL_Corners
        self.LibNames = []
        for corner in self.FEOL_Corners:
            self.LibNames.append("tcbn28hpcplusbwp7t40p140" + corner[0] + corner[1] + corner[2] + "c_ccs.lib")
    
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
    
    def _read_lib(self, libname):
        inlib = os.path.join(Global_var.Lib_Path, libname)
        cells = TimingLib_Parser.Read_TimingLib(inlib)
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingLib")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, libname.split(".")[0])
        with open(save_path, "wb") as f:
            pickle.dump(cells, f)
    
    def _read_saved_libs(self):
        print(f"Reading Saved Timing Libs Data")
        lib_data = []
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingLib")
        for lib in self.LibNames:
            save_path = os.path.join(save_dir, lib.split(".")[0])
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Timing library data of {lib} not found, using increLibs to read new libs.")
            with open(save_path, "rb") as f:
                lib_data.append(pickle.load(f))
        
        return lib_data
    
    """"""""""""""""""""""""""""""""""""
    """ Main function to process libs"""
    """"""""""""""""""""""""""""""""""""
    def lib_processer(self, reRead = True, increLibs = None):
        if reRead:
            if increLibs == None:
                AllLibs = self.LibNames
            else:
                AllLibs = increLibs
            
            all_done = threading.Event()
            status_thread = threading.Thread(target=self._output_status, args=(all_done, "Reading libs"))
            status_thread.start()
            
            try:
                max_processes = 8 # max number of processes
                processes = []
                for lib in AllLibs:
                    if len(processes) >= max_processes:
                        processes[0].join() # wait for the earliest process
                        processes.pop(0) # remove complete process
                        
                    process = Process(target=self._read_lib, args=(lib,))
                    processes.append(process)
                    process.start()
                    
                for process in processes:
                    process.join()  # wait for all process complete
                
                all_done.set()
            except Exception as e:
                print(f"Error occurred: {e}")
                all_done.set()
                raise
            status_thread.join()
        else:
            print(f"Using Saved Timing Libs.")
        
        lib_data = self._read_saved_libs()
        all_done = threading.Event()
        status_thread = threading.Thread(target=self._output_status, args=(all_done, "Processing libs"))
        status_thread.start()

        try:
            index1, index2, delay, trans =[], [], [], [] # index1, index2, delay, trans normalized seperately
            constrain_index1, constrian_index2, constrain = [], [], []
            index1_length, index2_length, delay_length, trans_length = [], [], [], []
            cons_index1_length, cons_index2_length, cons_length = [], [], []
            #print(lib_data[0]['CKAN2D0BWP7T40P140'])
            for cell in lib_data[0].values(): # lib_data[0] = [cell1, cell2 ...]
                for value in cell.delay.values():
                    index1.append(value.index1)
                    index2.append(value.index2)
                    delay.append(value.delay)
                    index1_length.append(len(value.index1))
                    index2_length.append(len(value.index2))
                    delay_length.append((len(value.index1), len(value.index2)))
                for value in cell.trans.values():
                    index1.append(value.index1)
                    index2.append(value.index2)
                    trans.append(value.trans)
                    index1_length.append(len(value.index1))
                    index2_length.append(len(value.index2))
                    trans_length.append((len(value.index1), len(value.index2)))
                for value in cell.constrain.values():
                    constrain_index1.append(value.index1)
                    constrian_index2.append(value.index2)
                    constrain.append(value.constrain)
                    cons_index1_length.append(len(value.index1))
                    cons_index2_length.append(len(value.index2))
                    cons_length.append((len(value.index1), len(value.index2)))
                for i in range(1, len(lib_data)):
                    #print(lib_data[i])
                    for value in lib_data[i][cell.name].delay.values():
                        index1.append(value.index1)
                        index2.append(value.index2)
                        delay.append(value.delay)
                        index1_length.append(len(value.index1))
                        index2_length.append(len(value.index2))
                        delay_length.append((len(value.index1), len(value.index2)))
                    for value in lib_data[i][cell.name].trans.values():
                        index1.append(value.index1)
                        index2.append(value.index2)
                        trans.append(value.trans)
                        index1_length.append(len(value.index1))
                        index2_length.append(len(value.index2))
                        trans_length.append((len(value.index1), len(value.index2)))
                    for value in lib_data[i][cell.name].constrain.values():
                        constrain_index1.append(value.index1)
                        constrian_index2.append(value.index2)
                        constrain.append(value.constrain)
                        cons_index1_length.append(len(value.index1))
                        cons_index2_length.append(len(value.index2))
                        cons_length.append((len(value.index1), len(value.index2)))
            #cell data are collected, next step normalization
            # flatten
            index1_flattened = [item for sublist in index1 for item in sublist]
            index2_flattened = [item for sublist in index2 for item in sublist]
            delay_flattened = [item for subdelay in delay for delay_line in subdelay for item in delay_line]
            trans_flattened = [item for subtrans in trans for trans_line in subtrans for item in trans_line]
            constrain_index1_flattened = [item for sublist in constrain_index1 for item in sublist]
            constrain_index2_flattened = [item for sublist in constrian_index2 for item in sublist]
            constrain_flattened = [item for subcons in constrain for cons_line in subcons for item in cons_line]
            # reshape
            index1_flattened = np.array(index1_flattened).reshape(-1,1)
            index2_flattened = np.array(index2_flattened).reshape(-1,1)
            delay_flattened = np.array(delay_flattened).reshape(-1,1)
            trans_flattened = np.array(trans_flattened).reshape(-1,1)
            constrain_index1_flattened = np.array(constrain_index1_flattened).reshape(-1,1)
            constrain_index2_flattened = np.array(constrain_index2_flattened).reshape(-1,1)
            constrain_flattened = np.array(constrain_flattened).reshape(-1,1)

            # min max scaling
            scaler = MinMaxScaler()
            index1_flattened = scaler.fit_transform(index1_flattened).squeeze().tolist()
            index2_flattened = scaler.fit_transform(index2_flattened).squeeze().tolist()
            delay_flattened = scaler.fit_transform(delay_flattened).squeeze().tolist()
            trans_flattened = scaler.fit_transform(trans_flattened).squeeze().tolist()
            constrain_index1_flattened = scaler.fit_transform(constrain_index1_flattened).squeeze().tolist()
            constrain_index2_flattened = scaler.fit_transform(constrain_index2_flattened).squeeze().tolist()
            constrain_flattened = scaler.fit_transform(constrain_flattened).squeeze().tolist()

            # reshape and storation
            index1_start, index1_end, index2_start, index2_end, delay_start, delay_end, trans_start, trans_end, count, count_delay, count_trans = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            cons_index1_start, cons_index1_end, cons_index2_start, cons_index2_end, cons_start, cons_end, count_cons_index, count_cons = 0, 0, 0, 0, 0, 0, 0, 0
            for cell in lib_data[0].values():
                for value in cell.delay.values():
                    index1_end, index2_end = index1_end + index1_length[count], index2_end + index2_length[count]
                    value.index1 = index1_flattened[index1_start:index1_end]
                    value.index2 = index2_flattened[index2_start:index2_end]
                    newdelay = []
                    for i in range(delay_length[count_delay][0]): # index1 row index2 column
                        delay_end += delay_length[count_delay][1]
                        newdelay.append(delay_flattened[delay_start:delay_end])
                        delay_start = delay_end
                    value.delay = newdelay
                    index1_start, index2_start = index1_end, index2_end
                    count += 1
                    count_delay += 1
                for value in cell.trans.values():
                    index1_end, index2_end = index1_end + index1_length[count], index2_end + index2_length[count]
                    value.index1 = index1_flattened[index1_start:index1_end]
                    value.index2 = index2_flattened[index2_start:index2_end]
                    newtrans = []
                    for i in range(trans_length[count_trans][0]):
                        trans_end += trans_length[count_trans][1]
                        newtrans.append(trans_flattened[trans_start:trans_end])
                        trans_start = trans_end
                    value.trans = newtrans
                    index1_start, index2_start = index1_end, index2_end
                    count += 1
                    count_trans += 1
                for value in cell.constrain.values():
                    cons_index1_end, cons_index2_end = cons_index1_end + cons_index1_length[count_cons_index], cons_index2_end + cons_index2_length[count_cons_index]
                    value.index1 = constrain_index1_flattened[cons_index1_start:cons_index1_end]
                    value.index2 = constrain_index2_flattened[cons_index2_start:cons_index2_end]
                    newcons = []
                    for i in range(cons_length[count_cons][0]):
                        cons_end += cons_length[count_cons][1]
                        newcons.append(constrain_flattened[cons_start:cons_end])
                        cons_start = cons_end
                    value.constrain = newcons
                    cons_index1_start, cons_index2_start = cons_index1_end, cons_index2_end
                    count_cons_index += 1
                    count_cons += 1
                for i in range(1, len(lib_data)):
                    for value in lib_data[i][cell.name].delay.values():
                        index1_end, index2_end = index1_end + index1_length[count], index2_end + index2_length[count]
                        value.index1 = index1_flattened[index1_start:index1_end]
                        value.index2 = index2_flattened[index2_start:index2_end]
                        newdelay = []
                        for _ in range(delay_length[count_delay][0]):
                            delay_end += delay_length[count_delay][1]
                            newdelay.append(delay_flattened[delay_start:delay_end])
                            delay_start = delay_end
                        value.delay = newdelay
                        index1_start, index2_start = index1_end, index2_end
                        count += 1
                        count_delay += 1
                    for value in lib_data[i][cell.name].trans.values():
                        index1_end, index2_end = index1_end + index1_length[count], index2_end + index2_length[count]
                        value.index1 = index1_flattened[index1_start:index1_end]
                        value.index2 = index2_flattened[index2_start:index2_end]
                        newtrans = []
                        for _ in range(trans_length[count_trans][0]):
                            trans_end += trans_length[count_trans][1]
                            newtrans.append(trans_flattened[trans_start:trans_end])
                            trans_start = trans_end
                        value.trans = newtrans
                        index1_start, index2_start = index1_end, index2_end
                        count += 1
                        count_trans += 1
                    for value in lib_data[i][cell.name].constrain.values():
                        cons_index1_end, cons_index2_end = cons_index1_end + cons_index1_length[count_cons_index], cons_index2_end + cons_index2_length[count_cons_index]
                        value.index1 = constrain_index1_flattened[cons_index1_start:cons_index1_end]
                        value.index2 = constrain_index2_flattened[cons_index2_start:cons_index2_end]
                        newcons = []
                        for _ in range(cons_length[count_cons][0]):
                            cons_end += cons_length[count_cons][1]
                            newcons.append(constrain_flattened[cons_start:cons_end])
                            cons_start = cons_end
                        value.constrain = newcons
                        cons_index1_start, cons_index2_start = cons_index1_end, cons_index2_end
                        count_cons_index += 1
                        count_cons += 1
            # padding
            # find max size of delay, trans and constrain template
            delay_index1_max, delay_index2_max, trans_index1_max, trans_index2_max, constrain_index1_max, constrain_index2_max = 0, 0, 0, 0, 0, 0
            for i in range(len(lib_data)):
                for cell in lib_data[i].values():
                    for delay in cell.delay.values():
                        if(len(delay.index1) > delay_index1_max):
                            delay_index1_max = len(delay.index1)
                        if(len(delay.index2) > delay_index2_max):
                            delay_index2_max = len(delay.index2)
                    for trans in cell.trans.values():
                        if(len(trans.index1) > trans_index1_max):
                            trans_index1_max = len(trans.index1)
                        if(len(trans.index2) > trans_index2_max):
                            trans_index2_max = len(trans.index2)
                    for constrains in cell.constrain.values():
                        if(len(constrains.index1) > constrain_index1_max):
                            constrain_index1_max = len(constrains.index1)
                        if(len(constrains.index2) > constrain_index2_max):
                            constrain_index2_max = len(constrains.index2)
            #padding
            for i in range(len(lib_data)):
                for cell in lib_data[i].values():
                    for delay in cell.delay.values():
                        for _ in range(delay_index1_max - len(delay.index1)):
                            # index1 padding
                            delay.index1.append(0)
                            #padding for delay
                            for line in delay.delay:
                                line.append(0)
                        # convert to max_index1 * max_index2
                        new_index1 = [[item] * delay_index2_max for item in delay.index1] # index1 * index2
                        delay.index1 = new_index1                
                        for j in range(delay_index2_max - len(delay.index2)):
                            delay.index2.append(0)
                            zero_list = [0 for _ in range(delay_index1_max)]
                            delay.delay.append(zero_list)
                        new_index2 = [delay.index2 for _ in range(delay_index1_max)] # index1 * index2
                        delay.index2 = new_index2
                    for trans in cell.trans.values():
                        for _ in range(trans_index1_max - len(trans.index1)):
                            # index1 padding
                            trans.index1.append(0)
                            #padding for delay
                            for line in trans.trans:
                                line.append(0)
                        # convert to max_index1 * max_index2
                        new_index1 = [[item] * trans_index2_max for item in trans.index1]
                        trans.index1 = new_index1                
                        for _ in range(trans_index2_max - len(trans.index2)):
                            trans.index2.append(0)
                            zero_list = [0 for _ in range(len(trans.index1))]
                            trans.trans.append(zero_list)
                        new_index2 = [trans.index2 for _ in range(trans_index1_max)]
                        trans.index2 = new_index2
                    for constrain in cell.constrain.values():
                        for _ in range(constrain_index1_max - len(constrain.index1)):
                            # index1 padding and convert to max_index1 * max_index2
                            constrain.index1.append(0)                
                            #padding for delay
                            for line in constrain.constrain:
                                line.append(0)
                        new_index1 = [[item] * constrain_index2_max for item in constrain.index1]
                        constrain.index1 = new_index1
                        for _ in range(constrain_index2_max - len(constrain.index2)):
                            constrain.index2.append(0)
                            zero_list = [0 for _ in range(len(constrain.index1))]
                            constrain.constrain.append(zero_list)
                        new_index2 = [constrain.index2 for _ in range(constrain_index1_max)]
                        constrain.index2 = new_index2
            
            self._save_normalized_data(lib_data)
            
            all_done.set()
            status_thread.join()
        except Exception as e:
            print(f"Error occurred: {e}")
            all_done.set()
            status_thread.join()
            raise

    def _save_normalized_data(self, lib_data):
        """Save the normalized data."""
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingLib_Normalized")
        os.makedirs(save_dir, exist_ok=True)
        for i, lib in enumerate(self.LibNames):
            save_path = os.path.join(save_dir, lib.split(".")[0] + "_normalized")
            with open(save_path, "wb") as f:
                pickle.dump(lib_data[i], f)
    
    def load_normalized_data(self):
        print(f"Loading timing lib data.")
        LibData = {}
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "TimingLib_Normalized")
        for corner in self.FEOL_Corners:
            lib = f"tcbn28hpcplusbwp7t40p140{corner[0]}{corner[1]}{corner[2]}c_ccs"
            save_path = os.path.join(save_dir, f"{lib}_normalized")
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Timing library data of {lib} not found, using increLibs to read new libs.")
            with open(save_path, "rb") as f:
                libdata = pickle.load(f)
            LibData[corner] = libdata
        return LibData

if __name__ == "__main__":
    LibProcesser = TimingLibTrans()
    LibProcesser.lib_processer(reRead=True)
