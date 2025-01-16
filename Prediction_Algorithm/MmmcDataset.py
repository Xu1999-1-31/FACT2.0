import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Mmmc_Dataset(Dataset):
    def __init__(self, encoded_data):
        self.corner = encoded_data[0]
        self.base_seq = encoded_data[1]
        self.target_seq = encoded_data[2]
        self.fpath = encoded_data[3]
        self.base_itf = encoded_data[4]
        self.pre_itf = encoded_data[5]
        self.slack = encoded_data[6]

    def __len__(self):
        return len(self.base_seq)

    def __getitem__(self, idx):
        corner = self.corner[idx]
        base_seq = self.base_seq[idx]
        target_seq = self.target_seq[idx]
        fpath =self.fpath[idx]
        base_itf = self.base_itf[idx]
        pre_itf = self.pre_itf[idx]
        slack = self.slack[idx]
        return corner, base_seq, target_seq, fpath, base_itf, pre_itf, slack

def collate_fn(batch):
    corner, base_seq, target_seq, fpath, base_itf, target_itf, slack = zip(*batch)
    padded_base_seq = pad_sequence(base_seq, batch_first=True, padding_value=0)
    padded_target_seq = pad_sequence(target_seq, batch_first=True, padding_value=0)
    padding_mask = (padded_base_seq == 0).all(dim=-1)
    fpath = torch.stack(fpath)
    base_itf = torch.stack(base_itf)
    target_itf = torch.stack(target_itf)
    slack = torch.stack(slack)
    return corner, padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack

if __name__ == "__main__":
    import TimingArc_Encoding
    from torch.utils.data import DataLoader
    dataset = TimingArc_Encoding.load_encoded_data("aes_cipher_top")
    # print(dataset[3])
    # for slack, fpath in zip(dataset[6], dataset[3]):
    #     print(slack, fpath)
    train_dataset = Mmmc_Dataset(dataset)
    #print(len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    #for batch in dataloader:
    #    print(len(batch))
    #for batch_idx, (padded_base_seq, padded_target_seq, padding_mask, fpath, base_itf, target_itf, slack) in enumerate(dataloader):
        # print(f"Batch {batch_idx + 1}:")
        # print("Padded Sequences:")
        # print(padded_base_seq)
        # print(padded_target_seq)
        # print("Padding Mask:")
        # print(padding_mask)
        # print("Slacks:")
        # print(slack)
        # print()
    
