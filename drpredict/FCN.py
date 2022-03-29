import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from modules import FullyConnectedModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI


class FCN(LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dims, **kwargs):
        self.network
        
def forward(self, x):
        batch_size = x.size()
        return out

    def training_step(self, batch, batch_idx):a
        return train_loss
    
    def training_epoch_end(self, outs):
        # log epoch metric
        train_acc_epoch = self.accuracy.compute()
        self.log('train_acc_epoch', train_acc_epoch, on_step=False, on_epoch=True)
        self.accuracy.reset()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch["sequence"], batch["reverse_complement"], batch["target"]
        outs = self(x, x_rev_comp).squeeze(dim=1)
        val_step_loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log("val_step_loss", val_step_loss)
        
        probs = torch.sigmoid(outs)
        self.auroc(probs, y.long())
        
        preds = torch.round(probs)
        val_acc_step = self.accuracy(preds, y.long())
        self.log("val_acc_step", val_acc_step)
        
    def validation_epoch_end(self, outs):
        # log epoch acc
        val_acc_epoch = self.accuracy.compute()
        self.log('val_acc_epoch', val_acc_epoch)
        self.accuracy.reset()
        
        # log epoch auroc
        val_auroc_epoch = self.auroc.compute()
        self.log("val_auroc_epoch", val_auroc_epoch, on_step=False, on_epoch=True)
        self.log("hp_metric", val_auroc_epoch, on_step=False, on_epoch=True)
        self.auroc.reset()
        
    def test_step(self, batch, batch_idx):
        x, x_rev_comp, y = batch["sequence"], batch["reverse_complement"], batch["target"]
        outs = self(x, x_rev_comp).squeeze(dim=1)
        test_loss = F.binary_cross_entropy_with_logits(outs, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
    
if __name__ == "__main__":
    cli = LightningCLI(FCN, MPRADataModule)