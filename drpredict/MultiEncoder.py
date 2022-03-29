# Will use pretrained encoder networks as starting point


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from modules import FullyConnectedModule


class MultiEncoder(LightningModule):
    def __init__(self, omics, input_dims, output_dims, hidden_dims=[], pretrained=False, encoder_kwargs=[], fcn_kwargs={}):
        super().__init__()
        self.pretrained = pretrained
        self.encoders = {}
        if pretrained:
            print("TODO!")
        else:
            for i in range(len(omics)):
                self.encoders[omics[i]] = FullyConnectedModule(input_dim=input_dims[i], output_dim=output_dims[i], hidden_dims=hidden_dims[i], **encoder_kwargs[i])
        self.encoding_len = np.sum(output_dims)
        self.fcn = FullyConnectedModule(input_dim=self.encoding_len, **fcn_kwargs)

    def forward(self, x):
        encoder_outs = {}
        for encoder in self.encoders:
            omic = encoder
            encoder_outs = encoder
        

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss