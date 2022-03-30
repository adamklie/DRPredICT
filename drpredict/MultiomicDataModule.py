import pytorch_lightning as pl
from MultiomicDataset import MultiomicDataset
from torch.utils.data import random_split, Subset, DataLoader

class MultiomicDataModule(pl.LightningDataModule):
    def __init__(self, file_ext: str, batch_size: int = 32, num_workers: int = 0, split=0.9, subset=False, dataset_kwargs={}):
        super().__init__()
        self.file_ext = file_ext
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.split=0.9
        self.subset = subset
        
    def setup(self, stage: str = None) -> None:
        dataset = MultiomicDataset(self.file_ext, **self.dataset_kwargs)
        dataset_len = len(dataset)
        if self.subset:
            self.train = Subset(dataset, indices=range(800))
            self.val = Subset(dataset, indices=range(801, 900, 1))
            self.test = Subset(dataset, indices=range(901, 1000, 1))
        else:
            train_len = int(dataset_len*self.split)
            test_len = dataset_len - train_len
            self.train, self.test = random_split(dataset, [train_len, test_len])
            val_len = int(dataset_len*0.2)
            train_len = train_len - val_len
            self.train, self.val = random_split(self.train, [train_len, val_len])

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
            self.test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers
        )