import torch
import torch.nn as nn
from pytorch_lightning.callbacks import BasePredictionWriter

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module: 'LightningModule', prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, "{}_predictions.pt".format(str(batch_idx))))
        torch.save(batch["auc"], os.path.join(self.output_dir, dataloader_idx, "{}_auc.pt".format(str(batch_idx))))

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))