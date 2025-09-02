import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np
import lightning.pytorch as pl
from GRALE.main import GRALE_model
import yaml
from GRALE.data.dataset import DataModule
import os 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

config = yaml.safe_load(open(f'GRALE/configs/dev.yaml', 'r'))
model = GRALE_model(config=config)
path_h5 = 'data/h5/PUBCHEM_16.h5'
datamodule = DataModule(
    path_h5=path_h5,
    batch_size=config['batchsize_effective'],
    n_data_epoch=config['n_data_epoch'],
    n_data_valid=config['n_data_valid']
)

def get_num_gpus():
    # USE
    if "SLURM_GPUS_ON_NODE" in os.environ:
        return int(os.environ.get("SLURM_GPUS_ON_NODE"))
    # Fallback: SLURM var
    return 1
n_gpus = get_num_gpus()

run_name = "test4"

# Explicit checkpoint callback
checkpoint_cb = ModelCheckpoint(
    dirpath=f"checkpoints/{run_name}",  # no conflict with logger
    filename="{epoch}-{edit_graph:.4f}",
    save_last=True,
    save_top_k=3,
    monitor="edit_graph",
    mode="min",
)

logger = WandbLogger(project="GRALE", 
                         name=run_name,
                         tags=[])
trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",  # or "auto"
        devices=n_gpus,
        strategy="ddp" if n_gpus > 1 else "auto",
        max_steps=config['n_grad_steps'],
        gradient_clip_val=config['max_grad_norm'],  
        gradient_clip_algorithm="norm",
        log_every_n_steps=100,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_cb]
    )
trainer.fit(model, datamodule=datamodule)
