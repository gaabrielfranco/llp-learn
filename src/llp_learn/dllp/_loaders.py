# Source: https://github.com/lucastassis/dllp/blob/main/loader.py
import numpy as np
from torch.utils.data import Dataset
    
# dataset class for regular llp datasets  
class LLPDataset(Dataset):
    def __init__(self, X, bags, proportions):
            self.X = X
            self.bags = bags
            self.proportions = proportions

    def __len__(self):
        return len(np.unique(self.bags))

    def __getitem__(self, idx):
        i = np.where(self.bags == idx)[0]
        X_batch = self.X[i]
        proportion = self.proportions[idx]
        return X_batch, proportion