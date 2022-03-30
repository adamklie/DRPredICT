import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

class MultiomicDataset(Dataset):
    """MultiomicDataset definition"""
    
    def __init__(self, file_ext, drug=False, mutation=False, expression=False, cn=False):
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
            
        if drug:
            self.drug_pairs = pd.read_table("%s_labels.tsv" % (file_ext), low_memory=False)
        
        #self.genes = {}
        
    def __len__(self):
        if "drug_pairs" in self.__dict__:
            return len(self.drug_pairs)
        else:
            if "mutation" in self.__dict__:
                return len(self.mutation)
            elif "expression" in self.__dict__:
                return len(self.expression)
            elif "cn" in self.__dict__:
                return len(self.cn)
    
    def __getitem__(self, idx):
        # Create a dictionary of different multiomic data slices (e.g. {"mutations": mut, "expression": exp...})
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {}
        if "drug_pairs" in self.__dict__:
            sample["drug_name"] = self.drug_pairs.Name.loc[idx]
            
            smile = self.drug_pairs.SMILE.loc[idx]
            morgan = AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smile), 2, nBits=2048)
            arr = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(morgan, arr)
            sample["drug_encoding"] = torch.Tensor(arr)
            
            sample["auc"] = torch.Tensor([self.drug_pairs.auc[idx]])
            sample["tissue"] = self.drug_pairs.lineage.loc[idx]
            cell_line = self.drug_pairs.CCLE_Name.loc[idx]
        else:
            if "mutation" in self.__dict__:
                cell_line = self.mutation.index[idx]
            elif "expression" in self.__dict__:
                cell_line = self.expression.index[idx]
            elif "cn" in self.__dict__:
                cell_line = self.cn.index[idx]

        sample["name"] = cell_line
        
        if "mutation" in self.__dict__:
            sample["mutation"] = torch.Tensor(self.mutation.loc[cell_line])
        if "expression" in self.__dict__:
            sample["expression"] = torch.Tensor(self.expression.loc[cell_line])
        if "cn" in self.__dict__:
            sample["cn"] = torch.Tensor(self.cn.loc[cell_line])

        return sample
    
