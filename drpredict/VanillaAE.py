import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from modules import FullyConnectedModule
from nn_utils import init_weights


class VanillaAE(LightningModule):
    def __init__(self, omic_type, input_dim, output_dim, hidden_dims=[], encoder_kwargs={}, decoder_kwargs={}):
        super().__init__()
        self.omic_type = omic_type
        self.encoder = FullyConnectedModule(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, **encoder_kwargs)
        self.decoder = FullyConnectedModule(input_dim=output_dim, output_dim=input_dim, hidden_dims=hidden_dims, **decoder_kwargs)
        init_weights(self.encoder)
        init_weights(self.decoder)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[self.omic_type]
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _common_step(self, batch, batch_idx, stage: str):
        x = batch[self.omic_type]
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss