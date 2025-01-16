import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import time
import threading
import torch
from tqdm import tqdm
import Global_var
import TrainDataPrepare

def TensorTrans(design):
    Data_Collector = TrainDataPrepare.Train_Data_Prepare()
    dataset = Data_Collector.load_dataset(design)
    corner, base_delay, base_trans, base_constrain, target_delay, target_trans, target_constrain, fpaths, base_itf, target_itf, slack_delay = [], [], [], [], [], [], [], [], [], [], []
    for i in tqdm(range(len(dataset)), desc=f"Processing Data in {design}"):
        data = dataset[i]
        if 'm' in data[0][2]:
            fpath = data[7] + [-40/165]
        else:
            fpath = data[7] + [float(data[0][2])/165]
        corner.append(data[0]); base_delay.append(torch.tensor(data[1])); base_trans.append(torch.tensor(data[2])); base_constrain.append(torch.tensor(data[3]))
        target_delay.append(torch.tensor(data[4])); target_trans.append(torch.tensor(data[5])); target_constrain.append(torch.tensor(data[6]))
        fpaths.append(torch.tensor(fpath)); base_itf.append(torch.tensor(data[8]).view(-1)); target_itf.append(torch.tensor(data[9]).view(-1));
        slack_delay.append(torch.tensor(data[10]))
    Save_Path = Global_var.Dataset_Path
    save_dir = os.path.join(Save_Path, "NonEncoded_Dataset")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{design}_NonEncoded.pt")
    torch.save((corner, base_delay, base_trans, base_constrain, target_delay, target_trans, target_constrain, fpath, base_itf, target_itf, slack_delay), save_path)

def output_status(all_done, status):
        dot_count = 0
        while not all_done.is_set():
            dot_count = dot_count + 1 if dot_count < 6 else 0
            if(dot_count == 0):
                print(f"\r{status} " + " " * 100, end="")
            else:
                print(f"\r{status} " + "." * dot_count, end="")
                time.sleep(1.5)
        print(f"\r{status} " + "." * 6 + " " * 100, end="")
        sys.stdout.flush()
        print()

def load_encoded_data(design):
    # print(f"Loading encoded data of {design}.")
    all_done = threading.Event()
    status_thread = threading.Thread(target=output_status, args=(all_done, f"Loading encoded data of {design}"))
    status_thread.start()
    try:
        Save_Path = Global_var.Dataset_Path
        save_dir = os.path.join(Save_Path, "NonEncoded_Dataset")
        save_path = os.path.join(save_dir, f"{design}_NonEncoded.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"NonEncoded data of {design} not found, using TensorTrans to rebuild data.")
        dataset = torch.load(save_path, weights_only=True)
        all_done.set()
    except Exception as e:
        print(f"Error occurred: {e}")
        all_done.set()
        raise
    status_thread.join()
    return dataset

if __name__ == "__main__":
    TensorTrans("aes_cipher_top")