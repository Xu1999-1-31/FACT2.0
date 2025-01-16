import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import Global_var
from MmmcDataset import Mmmc_Dataset, collate_fn
from Dynamic_LR import DynamicLearningRateScheduler
import TimingArc_Encoding
from sklearn.metrics import r2_score
import model

def set_seed(seed=42):
    # Set the seed for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        # torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior
    
    print(f"Seed set to {seed}")

def TrainPredictor():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Train_Dataset = [[], [], [], [], [], [], []]
    for design in Global_var.TrainDesigns:
        dataset = TimingArc_Encoding.load_encoded_data(design)
        for i in range(len(Train_Dataset)):
            Train_Dataset[i] += dataset[i]
    
    batch_size = 1024
    Train_Dataset = Mmmc_Dataset(Train_Dataset)
    Dataloader = DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # model parameters
    # d_model = 128
    # nhead = 8
    # num_encoder_layers = 3
    # num_decoder_layers = 3
    # MMMCPredictor = model.MMMC_Transformer().to(device)
    # MMMCPredictor = model.Res_MMMC_Transformer().to(device)
    # MMMCPredictor = model.MMMC_BidTransformer().to(device)
    # MMMCPredictor = model.Res_MMMC_BidTransformer().to(device)
    # MMMCPredictor = model.Cross_Head_MMMC_Transformer().to(device)
    MMMCPredictor = model.MmmcTransformerDynamicBidirectional().to(device)
    MMMCPredictor.train()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(MMMCPredictor.parameters(), lr=0.0005)
    
    # Learning rate scheduler: ReduceLROnPlateau based on loss
    #lr_scheduler = DynamicLearningRateScheduler(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, verbose=True)
    min_lr = 0.00005
    
    num_epochs = 200
    for epoch in range(num_epochs):
        loss_item, count = 0, 0
        pred, truth = [], []
        for batch in Dataloader:
            _, padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack_delay = batch
            padded_base_seq = padded_base_seq.to(device); padded_target_seq = padded_target_seq.to(device)
            padding_mask = padding_mask.to(device); fpath = fpath.to(device); base_itf = base_itf.to(device)
            target_itf = target_itf.to(device); slack_delay = slack_delay.to(device)
            pre_slack_delay = MMMCPredictor(padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf)
            loss = criterion(pre_slack_delay, slack_delay)
            
            pred.append(pre_slack_delay.detach().cpu().numpy())
            truth.append(slack_delay.detach().cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item += loss.item()
            count += 1
        
        if epoch==0 or epoch%20 == 19:
            pred = np.concatenate(pred, axis=0)
            truth = np.concatenate(truth, axis=0)
            r2 = r2_score(truth, pred)
            print(f"R2 of training set:", r2)
        
        lr_scheduler.step(loss_item/count)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            optimizer.param_groups[0]['lr'] = min_lr
            current_lr = min_lr
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_item/count:.10f}, Learning Rate: {current_lr:.5f}")
    
    MMMCPredictor.eval()
    save_dir = Global_var.Model_Path
    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"MmmcPredictor.pt")
    # save_path = os.path.join(save_dir, f"MmmcPredictor_res.pt")
    # save_path = os.path.join(save_dir, f"MmmcPredictor_ch.pt")
    # save_path = os.path.join(save_dir, f"MmmcPredictor_bid.pt")
    # save_path = os.path.join(save_dir, f"MmmcPredictor_res_bid.pt")
    save_path = os.path.join(save_dir, f"MmmcPredictor_dybid.pt")
    torch.save(MMMCPredictor.state_dict(), save_path)
    
    # test prediction
    # Total_pre, Total_golden = [], []
    # Dataloader = DataLoader(Train_Dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
    # for batch in Dataloader:
    #     padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack_delay = batch
    #     padded_base_seq = padded_base_seq.to(device); padded_target_seq = padded_target_seq.to(device)
    #     padding_mask = padding_mask.to(device); fpath = fpath.to(device); base_itf = base_itf.to(device)
    #     target_itf = target_itf.to(device)
    #     pre_slack_delay = MMMCPredictor(padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf)
    #     Total_pre.append(pre_slack_delay.cpu().detach().numpy())
    #     Total_golden.append(slack_delay.cpu().detach().numpy())
    
    # Total_pre = np.concatenate(Total_pre, axis=0)
    # Total_golden = np.concatenate(Total_golden, axis=0)
    
    # r2 = r2_score(Total_golden, Total_pre)
    # print(f"R2 of training set:", r2)

def loadPredictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Global_var.Model_Path
    save_path = os.path.join(save_dir, f"MmmcPredictor_dybid.pt")
    # MMMCPredictor = model.MMMC_Transformer()
    # MMMCPredictor = model.Res_MMMC_Transformer()
    # MMMCPredictor = model.MMMC_BidTransformer()
    # MMMCPredictor = model.Res_MMMC_BidTransformer()
    # MMMCPredictor = model.Cross_Head_MMMC_Transformer()
    MMMCPredictor = model.MmmcTransformerDynamicBidirectional()
    MMMCPredictor.load_state_dict(torch.load(save_path, weights_only=True))
    MMMCPredictor.eval().to(device)
    
    return MMMCPredictor

def TestModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MMMCPredictor = loadPredictor()
    
    with torch.no_grad():
        # Training set R2 score
        Dataset = [[], [], [], [], [], [], []]
        for design in Global_var.TrainDesigns:
            dataset = TimingArc_Encoding.load_encoded_data(design)
            for i in range(len(Dataset)):
                Dataset[i] += dataset[i]
        Tensor_Dataset = Mmmc_Dataset(Dataset)
        Dataloader = DataLoader(Tensor_Dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
        pred, truth = [], []
        for batch in Dataloader:
            corner, padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack_delay = batch
            padded_base_seq = padded_base_seq.to(device); padded_target_seq = padded_target_seq.to(device)
            padding_mask = padding_mask.to(device); fpath = fpath.to(device); base_itf = base_itf.to(device)
            target_itf = target_itf.to(device); slack_delay = slack_delay.to(device)
            
            pre_slack_delay = MMMCPredictor(padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf)
            pred.append(pre_slack_delay.detach().cpu().numpy())
            truth.append(slack_delay.detach().cpu().numpy())
        pred = np.concatenate(pred, axis=0)
        truth = np.concatenate(truth, axis=0)
        r2 = r2_score(truth, pred)
        print(f"R2 value of train set:", r2)
        
        # Test set R2 score
        Dataset = [[], [], [], [], [], [], []]
        Datasets = []
        for design in Global_var.TestDesigns:
            dataset = TimingArc_Encoding.load_encoded_data(design)
            Datasets.append(dataset)
        
        for design, dataset in zip(Global_var.TestDesigns, Datasets):
            Tensor_Dataset = Mmmc_Dataset(dataset)
            Dataloader = DataLoader(Tensor_Dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
            pred, truth = [], []
            for batch in Dataloader:
                corner, padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack_delay = batch
                padded_base_seq = padded_base_seq.to(device); padded_target_seq = padded_target_seq.to(device)
                padding_mask = padding_mask.to(device); fpath = fpath.to(device); base_itf = base_itf.to(device)
                target_itf = target_itf.to(device); slack_delay = slack_delay.to(device)
                
                pre_slack_delay = MMMCPredictor(padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf)
                pred.append(pre_slack_delay.detach().cpu().numpy())
                truth.append(slack_delay.detach().cpu().numpy())
            pred = np.concatenate(pred, axis=0)
            truth = np.concatenate(truth, axis=0)
            r2 = r2_score(truth, pred)
            print(f"R2 value of {design}:", r2)
        
        for dataset in Datasets:
            for i in range(len(Dataset)):
                Dataset[i] += dataset[i]
        
        Tensor_Dataset = Mmmc_Dataset(Dataset)
        Dataloader = DataLoader(Tensor_Dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
        pred, truth = [], []
        for batch in Dataloader:
            corner, padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack_delay = batch
            padded_base_seq = padded_base_seq.to(device); padded_target_seq = padded_target_seq.to(device)
            padding_mask = padding_mask.to(device); fpath = fpath.to(device); base_itf = base_itf.to(device)
            target_itf = target_itf.to(device); slack_delay = slack_delay.to(device)
            
            pre_slack_delay = MMMCPredictor(padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf)
            pred.append(pre_slack_delay.detach().cpu().numpy())
            truth.append(slack_delay.detach().cpu().numpy())
        pred = np.concatenate(pred, axis=0)
        truth = np.concatenate(truth, axis=0)
        r2 = r2_score(truth, pred)
        print(f"R2 value of test set:", r2)
        

if __name__ == "__main__": 
    # TrainPredictor()
    TestModel()
