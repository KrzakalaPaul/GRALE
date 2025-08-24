from lightning.pytorch.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch):
        print(batch)
        train_loss = 0*self.linear(batch).sum() + len(batch)
        self.log('train_loss', train_loss, on_epoch=True, on_step=True, batch_size=len(batch))
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Dummy dataset
x = torch.randn(100, 10)
train_loader = DataLoader(x, batch_size=32)
model = LitModel()
wandb_logger = WandbLogger(project="GRALE", save_dir="logs")
trainer = pl.Trainer(max_epochs=5, accelerator='cuda', devices=1, logger=wandb_logger)
trainer.fit(model, train_loader)