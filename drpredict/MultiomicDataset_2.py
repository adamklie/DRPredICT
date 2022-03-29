import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiomicDataset(Dataset):
    """MultiomicDataset definition"""
    
    def __init__(self, file_ext, mutation=True, expression=True, cn=True):
        """
        Args:
            file_ext (string): 
        """
        self.file_ext = file_ext
        if mutation:
            self.mutation = pd.read_csv("%s_mutation.tsv" % (file_ext), index_col=0)
        if expression:
            self.expression = pd.read_csv("%s_expression.tsv" % (file_ext), index_col=0)
        if cn:
            self.cn = pd.read_csv("%s_cn.tsv" % (file_ext), index_col=0)
        temp = pd.read_csv("%s_labels.tsv" % (file_ext), index_col=0)
        self.drug_pairs = temp[['']]
        #TODO read in multiomic files
        # Read in each multiomic file specified
        
    def __len__(self):
        return len(self.drug_pairs)
    
    def __getitem__(self, idx):
        # Create a dictionary of different multiomic data slices (e.g. {"mutations": mut, "expression": exp...})
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {}
        if mutation:
            sample["mutation"] = self.mutation[idx]
        if expression:
            sample["expression"] = self.expression[idx]
        if cn:
            sample["cn"] = self.cn[idx]
        
        #sample = {"sequence": seq, "target": target}
        
        # TODOD apply transforms to the sample if desired
        if self.transform:
            sample = self.transform(sample)
        return sample
    
