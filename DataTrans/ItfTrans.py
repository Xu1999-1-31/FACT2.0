import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import Itf_Parser
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

class ItfTrans():
    def __init__(self):
        self.BEOL_Corners = Global_var.BEOL_Corners
    
    def _read_itf(self, BEOL):
        initf = os.path.join(Global_var.Itf_Path, f"cln28hpc+_1p09m+ut-alrdl_4x2y2r_{BEOL}.itf")
        metals = Itf_Parser.Read_Itf(initf)
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "ITF")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{BEOL}_itf")
        with open(save_path, "wb") as f:
            pickle.dump(metals, f)
    
    def _read_saved_itf(self):
        ALL_ITF = {}
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "ITF")
        for BEOL in self.BEOL_Corners:
            save_path = os.path.join(save_dir, f"{BEOL}_itf")
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"ITF data of {BEOL} not found, using reRead to update itfs.")
            with open(save_path, "rb") as f:
                metals = pickle.load(f)
                ALL_ITF[BEOL] = metals
        return ALL_ITF
    
    def itf_processer(self, reRead=True):
        if reRead:
            for BEOL in self.BEOL_Corners:
                print(f"Read {BEOL} itf data.")
                self._read_itf(BEOL)
        else:
            print(f"Using Saved ITF data.")
        All_ITF = self._read_saved_itf()
        ITF_data = []
        for BEOL, metals in All_ITF.items():
            for metal in metals.values():
                ITF_data.append([metal.thickness, metal.e_above, metal.e_below])
        scaler = MinMaxScaler()
        ITF_data = scaler.fit_transform(np.array(ITF_data)).tolist()
        Itf_data = {}
        count = 0
        for BEOL in self.BEOL_Corners:
            itf_data = []
            for i in range(len(metals)):
                itf_data.append(ITF_data[count])
                count += 1
            Itf_data[BEOL] = itf_data
        
        self._save_normalized_data(Itf_data)
        
    def _save_normalized_data(self, Itf_data):
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "ITF_Normalized")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"Itf_data")
        with open(save_path, "wb") as f:
            pickle.dump(Itf_data, f)
    
    def load_normalized_data(self):
        print(f"Loading itf data.")
        Save_Path = Global_var.Saved_Data_Path
        save_dir = os.path.join(Save_Path, "ITF_Normalized")
        save_path = os.path.join(save_dir, f"Itf_data")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Normalized ITF data not found, using itf_processer to update itfs.")
        with open(save_path, "rb") as f:
            Itf_data = pickle.load(f)
        return Itf_data

if __name__ == "__main__":
    ItfProcesser = ItfTrans()
    ItfProcesser.itf_processer(reRead=True)