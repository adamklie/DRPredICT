import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from modules import FullyConnectedModule
from nn_utils import init_weights
from MultiomicDataModule import MultiomicDataModule
from pytorch_lightning.utilities.cli import LightningCLI


class MultiEncoder(LightningModule):
    """MultiomEncoder class definition.
       Defines a multi-modal model that is capable of taking in multiple input types and encode each of them
       with a separate FCN. FCN's are built from fully connected networks (FullyConnectedModule class). Keep the order
       of initialization arguments the same for every list
    """
    def __init__(self, omics, input_dims, output_dims, hidden_dims=[], pretrained=[], encoder_kwargs=[], fcn_kwargs={}):
        """
        Args:
            omics (list of strings): defines the omics types to include (eventually want this to be the paths for file names), e.g. ["mutation", "expression"]
            input_dims (list-like): The input dimensions of each encoder to train, e.g. [2000, 2000, 2048]
            output_dims (list-like): The output dimensions of each encoder to train, e.g. [2000, 2000, 2048]
            hidden_dims (list-like of list like, optional): The hidden dims (as lists) for each encoder, e.g. [[2000, 2000, 2048], [2000, 2000, 2048]]
            pretrained (list of nn.Modules, optional): List of pretrained models to use (will add these to model architecture)
            encoder_kwargs (list of dicts, optional): List of dictionaries with any other keyword arguments to be passed to each encoder
            fcn_kwargs (dict, optional): Dictionary defining keyword arguments to be passed to the fully connected layer that takes in concatenated encodings
        """
        super().__init__()
        self.omics = omics
        self.pretrained = pretrained
        self.encoders = nn.ModuleDict()
        for model in pretrained:
            print("TODO!")
        else:
            for i in range(len(omics)):
                self.encoders[omics[i]] = FullyConnectedModule(input_dim=input_dims[i], output_dim=output_dims[i], hidden_dims=hidden_dims[i], **encoder_kwargs[i])
        self.encoding_len = np.sum(output_dims)
        self.fcn = FullyConnectedModule(input_dim=self.encoding_len, **fcn_kwargs)
        self.save_hyperparameters()

    def forward(self, x):
        encoder_outs = []
        for i, omic in enumerate(self.encoders.keys()):
            encoder = self.encoders[omic]
            encoder_outs.append(encoder(x[omic]))
        out = torch.cat(encoder_outs, dim=1)
        out = self.fcn(out)
        return out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
        
    def predict_step(self, batch, batch_idx):
        x, y = batch, batch["auc"]
        self.eval()
        pred = self(x)
        return pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch, batch["auc"]
        preds = self(x)
        loss = F.mse_loss(y, preds)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss
    
    
if __name__ == "__main__":
    cli = LightningCLI(MultiEncoder, MultiomicDataModule)