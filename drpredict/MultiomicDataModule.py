import pytorch_lightning as pl
from MultiomicDataset import MultiomicDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
#from transforms import TODO add potential transforms
#from load_data import load

class MultiomicDataModule(pl.LightningDataModule):
    def __init__(self, file_ext: str, batch_size: int = 32, num_workers: int = 0, split=0.9, dataset_kwargs={}):
        super().__init__()
        self.file_ext = file_ext
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers
        self.dataset_kwargs = **dataset_kwargs
        
    def setup(self, stage: str = None) -> None:
        #TODO: load file
        dataset = MultiomicDataset(self.file_ext, self.dataset_kwargs)
        
        # TODO make for train, val and test split
        dataset_len = len(dataset)
        train_len = int(dataset_len*self.split)
        test_len = dataset_len - train_len
        self.train, self.test = random_split(dataset, [train_len, val_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers
        )