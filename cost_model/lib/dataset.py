import numpy as np
import torch
from torch.utils.data import Dataset

class MTDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = torch.from_numpy(data).float()
        if label is not None:
            self.label = torch.from_numpy(label).float().unsqueeze(1)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return self.data.shape[0]
