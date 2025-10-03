import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self, dense_tensor, fix_sparse_tensor, varlen_tensor, targets_tensor=None, sample_weights_tensor=None):
        self.dense_tensor = dense_tensor
        self.fix_sparse_tensor = fix_sparse_tensor
        self.varlen_tensor = varlen_tensor
        self.targets_tensor = targets_tensor
        self.sample_weights_tensor = sample_weights_tensor

    def __len__(self):
        return len(self.dense_tensor)

    def __getitem__(self, idx):
        items = [self.dense_tensor[idx], self.fix_sparse_tensor[idx], self.varlen_tensor[idx]]
        if self.targets_tensor is not None:
            items.append(self.targets_tensor[idx])
        if self.sample_weights_tensor is not None:
            items.append(self.sample_weights_tensor[idx])
        return tuple(items)
    