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
            self.mutation = pd.read_table("%s_mutations.tsv" % (file_ext), index_col=0)
        if expression:
            self.expression = pd.read_table("%s_expression.tsv" % (file_ext), index_col=0)
        if cn:
            self.cn = pd.read_table("%s_cn.tsv" % (file_ext), index_col=0)
        self.drug_pairs = pd.read_table("%s_labels.tsv" % (file_ext))
        
        #TODO read in multiomic files
        # Read in each multiomic file specified
        
    def __len__(self):
        return len(self.drug_pairs)
    
    def __getitem__(self, idx):
        # Create a dictionary of different multiomic data slices (e.g. {"mutations": mut, "expression": exp...})
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Extract cell line name from cell line - drug pairs
        cell_line = self.drug_pairs.CCLE_Name.loc[idx]
        
        sample = {}
        sample["name"] = cell_line
        sample["tissue"] = self.drug_pairs.lineage.loc[idx]
        sample["drug_name"] = self.drug_pairs.Name.loc[idx]
        sample["drug_encoding"] = self.drug_pairs.SMILE.loc[idx]
        if self.mutation:
            sample["mutation"] = torch.Tensor(self.mutation.loc[cell_line])
        if expression:
            sample["expression"] = torch.Tensor(self.expression.loc[cell_line])
        if cn:
            sample["cn"] = torch.Tensor(self.cn.loc[cell_line])
        
        #sample = {"sequence": seq, "target": target}
        
        # TODOD apply transforms to the sample if desired
        if self.transform:
            sample = self.transform(sample)
        return sample
    
