from GRALE.utils.dataset import MyDataModule
import pytorch_lightning as pl
import torch.nn as nn
import torch

path_h5 = 'data/h5/PUBCHEM_dev.h5'
datamodule = MyDataModule(path_h5=path_h5, batch_size=32, n_data_epoch=1000, n_data_valid=500)

class LitModel(pl.LightningModule):
    def __init__(self, n_node_labels):
        super().__init__()
        self.encoder = nn.Linear(n_node_labels, 1)
        
    def forward(self, inputs):
        x = self.encoder(inputs)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch):
        batch_size = batch.nodes.labels.shape[0]
        y = self(batch.nodes.labels)
        loss = y.sum()
        self.log('train_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch):
        batch_size = batch.nodes.labels.shape[0]
        y = self(batch.nodes.labels)
        loss = y.sum()
        self.log('val_loss', loss, on_epoch=True, batch_size=batch_size)

model = LitModel(n_node_labels=datamodule.metadata['n_node_labels'])

trainer = pl.Trainer(max_epochs=5, accelerator='cuda', devices=1, reload_dataloaders_every_n_epochs=1)

trainer.fit(model, datamodule=datamodule)