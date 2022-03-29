import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiomicDataset(Dataset):
    """MultiomicDataset definition"""
    
    def __init__(self, file_ext):
        """
        Args:
            file_ext (string): 
        """
        self.file_ext = file_ext
        #TODO read in multiomic files
        
    def __len__(self):
        return
    
    def __getitem__(self, idx):
        # Create a dictionary of different multiomic data slices (e.g. {"mutations": mut, "expression": exp...})
        if torch.is_tensor(idx):
            idx = idx.tolist()


        #sample = {"sequence": seq, "target": target}
        
        # TODOD apply transforms to the sample if desired
        if self.transform:
            sample = self.transform(sample)
        return sample