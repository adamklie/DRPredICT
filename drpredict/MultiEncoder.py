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
    def __init__(self, omics, input_dims, output_dims, hidden_dims=[], pretrained=False, encoder_kwargs=[], fcn_kwargs={}):
        super().__init__()
        self.omics = omics
        self.pretrained = pretrained
        self.encoders = nn.ModuleDict()
        if pretrained:
            print("TODO!")
        else:
            for i in range(len(omics)):
                self.encoders[omics[i]] = FullyConnectedModule(input_dim=input_dims[i], output_dim=output_dims[i], hidden_dims=hidden_dims[i], **encoder_kwargs[i])
                init_weights(self.encoders[omics[i]])
        self.encoding_len = np.sum(output_dims)
        self.fcn = FullyConnectedModule(input_dim=self.encoding_len, **fcn_kwargs)
        init_weights(self.fcn)
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

from pytorch_lightning.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module: 'LightningModule', prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, "{}_predictions.pt".format(str(batch_idx))))
        torch.save(batch["auc"], os.path.join(self.output_dir, dataloader_idx, "{}_auc.pt".format(str(batch_idx))))

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
    
if __name__ == "__main__":
    cli = LightningCLI(MultiEncoder, MultiomicDataModule)