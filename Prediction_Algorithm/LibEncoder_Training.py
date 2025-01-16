import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import model
import torch
from torch.utils.data import TensorDataset, DataLoader
import TimingLibTrans

def TrainLibEncoder(Encoder_type):
    encoder_map = {
        "Delay": lambda cell: [(delay.index1, delay.index2, delay.delay) for delay in cell.delay.values()],
        "Trans": lambda cell: [(trans.index1, trans.index2, trans.trans) for trans in cell.trans.values()],
        "Cons": lambda cell: [(cons.index1, cons.index2, cons.constrain) for cons in cell.constrain.values()]
    }
    
    autoencoder_map = {
        "Delay": model.Delay_Autoencoder,
        "Trans": model.Trans_Autoencoder,
        "Cons": model.Cons_Autoencoder
    }
    
    if Encoder_type not in encoder_map:
        raise ValueError("Encoder_type should be one of 'Delay', 'Trans' or 'Cons'!")
    
    dataset = []
    LibTrans = TimingLibTrans.TimingLibTrans()
    Lib_data = LibTrans.load_normalized_data()
    
    print(f"Processing Lib {Encoder_type} data.")
    for cells in Lib_data.values():
        for cell in cells.values():
            dataset.extend(encoder_map[Encoder_type](cell))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.tensor(dataset, dtype=torch.float32).to(device)
    TrainDataset = TensorDataset(dataset, dataset)
    batch_size = 128
    data_loader = DataLoader(TrainDataset, batch_size = batch_size, shuffle = True)
    
    autoEncoder = autoencoder_map[Encoder_type]().to(device).train()
    optimizer = torch.optim.Adam(autoEncoder.parameters(), lr = 0.0001)
    criterion = torch.nn.MSELoss()
    num_epochs = 500
    print(f"Training {Encoder_type} autoEncoder.")
    for epoch in range(num_epochs):
        loss_item = 0
        count = 0
        for data, _ in data_loader:
            data_reconstructed = autoEncoder(data)
            loss = criterion(data_reconstructed, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item += loss.item()
            count += 1
        print(f'Epoch [{epoch+1}/{num_epochs}], {Encoder_type} Loss: {loss_item/count:.10f}')
    autoEncoder.eval()
    save_dir = Global_var.Model_Path
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{Encoder_type}_autoEncoder.pt")
    torch.save(autoEncoder.state_dict(), save_path)

def loadEncoder(device):
    save_dir = Global_var.Model_Path
    encoder_map = {
        "Delay": model.Delay_Autoencoder,
        "Trans": model.Trans_Autoencoder,
        "Cons": model.Cons_Autoencoder
    }
    encoders = []

    for name in encoder_map:
        save_path = os.path.join(save_dir, f"{name}_autoEncoder.pt")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Encoder of {name} not found, using TrainLibEncoder to retrain it.")
        autoEncoder = encoder_map[name]()
        autoEncoder.load_state_dict(torch.load(save_path, weights_only=True))
        autoEncoder.eval().to(device)
        encoders.append(autoEncoder)
    
    return tuple(encoders)
    
    
if __name__ == "__main__": 
    TrainLibEncoder("Delay")