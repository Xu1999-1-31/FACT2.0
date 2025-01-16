import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from tqdm import tqdm
import psutil
import torch
import pickle
import time
import threading
import Global_var
import TrainDataPrepare
import OptimizeDataPrepare
import LibEncoder_Training

def TimingArc_Encoder(design, base_corner=Global_var.base_corner, optimize=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delay_encoder, trans_encoder, cons_encoder = LibEncoder_Training.loadEncoder(device)

    output_frequency = 100000
    if optimize:
        Data_Collector = OptimizeDataPrepare.Optimize_Data_Prepare()
        dataset = Data_Collector.load_dataset(design)
    else:
        Data_Collector = TrainDataPrepare.Train_Data_Prepare()
        dataset = Data_Collector.load_dataset(design)
    corner, base_seqs, target_seqs, fpaths, base_itf, target_itf, slack_delay = [], [], [], [], [], [], []
    for i in tqdm(range(len(dataset)), desc=f"Processing Data in {design}"):
        data = dataset[i]
        
        # timing arc encoding using LibEncoder
        with torch.no_grad():
            base_seq = torch.cat([
                delay_encoder.encoder(torch.tensor(data[1]).to(device)),
                trans_encoder.encoder(torch.tensor(data[2]).to(device))
            ], dim=1)
            
            target_seq = torch.cat([
                delay_encoder.encoder(torch.tensor(data[4]).to(device)),
                trans_encoder.encoder(torch.tensor(data[5]).to(device))
            ], dim=1)
        
            base_seq = torch.cat([base_seq, cons_encoder.encoder(torch.tensor(data[3]).to(device).unsqueeze(0))], dim=0)
            target_seq = torch.cat([target_seq, cons_encoder.encoder(torch.tensor(data[6]).to(device).unsqueeze(0))], dim=0)
        
        base_seq = base_seq.to("cpu")
        target_seq = target_seq.to("cpu")
        
        # temperature
        if 'm' in data[0][2]:
            fpath = data[7] + [-40/165]
        else:
            fpath = data[7] + [float(data[0][2])/165]
        
        corner.append(data[0]); base_seqs.append(base_seq); target_seqs.append(target_seq); fpaths.append(torch.tensor(fpath))
        base_itf.append(torch.tensor(data[8]).view(-1)); target_itf.append(torch.tensor(data[9]).view(-1)); slack_delay.append(torch.tensor(data[10]))
        if (i + 1) % output_frequency == 0:
            memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
            tqdm.write(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Saving Encoded Dataset.")
    Save_Path = Global_var.Dataset_Path
    if optimize:
        save_dir = os.path.join(Save_Path, f"Encoded_Optimization_Dataset/{base_corner[0]}_{base_corner[1]}_{base_corner[2]}_{base_corner[3]}")
    else:
        save_dir = os.path.join(Save_Path, f"Encoded_Dataset/{base_corner[0]}_{base_corner[1]}_{base_corner[2]}_{base_corner[3]}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{design}_encoded.pt")
    torch.save((corner, base_seqs, target_seqs, fpaths, base_itf, target_itf, slack_delay), save_path)
    # with open(save_path, "wb") as f:
    #     pickle.dump((corner, base_seqs, target_seqs, fpaths, base_itf, target_itf, slack_delay), f)

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

def load_encoded_data(design, base_corner=Global_var.base_corner, optimize=False):
    # print(f"Loading encoded data of {design}.")
    all_done = threading.Event()
    status_thread = threading.Thread(target=output_status, args=(all_done, f"Loading encoded data of {design}"))
    status_thread.start()
    try:
        Save_Path = Global_var.Dataset_Path
        if optimize:
            save_dir = os.path.join(Save_Path, f"Encoded_Optimization_Dataset/{base_corner[0]}_{base_corner[1]}_{base_corner[2]}_{base_corner[3]}")
        else:
            save_dir = os.path.join(Save_Path, f"Encoded_Dataset/{base_corner[0]}_{base_corner[1]}_{base_corner[2]}_{base_corner[3]}")
        save_path = os.path.join(save_dir, f"{design}_encoded.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Encoded data of {design} not found, using TimingArc_Encoder to rebuild data.")
        dataset = torch.load(save_path, weights_only=True)
        all_done.set()
    except Exception as e:
        print(f"Error occurred: {e}")
        all_done.set()
        raise
    status_thread.join()
    return dataset

if __name__ == "__main__":
    for design in Global_var.Designs:
        TimingArc_Encoder(design)
    # TimingArc_Encoder("des")
    # TimingArc_Encoder("aes_cipher_top", optimize=True)
