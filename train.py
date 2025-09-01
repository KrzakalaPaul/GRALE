import pytorch_lightning as pl
from GRALE.data.dataset import DataModule
from GRALE.main import GRALE_model
import argparse
import os
import yaml
from lightning.pytorch.loggers import WandbLogger

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dev')
    parser.add_argument('--run_name', type=str, default='dev')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'GRALE/configs/{args.config}.yaml', 'r'))
    run_name = args.run_name
    return config, run_name

def get_model(config):
    model = GRALE_model(config=config)
    return model

def get_data(config):
    path_h5 = 'data/h5/PUBCHEM_dev.h5'
    datamodule = DataModule(
        path_h5=path_h5,
        batch_size=config['batchsize'],
        n_data_epoch=config['n_data_epoch'],
        n_data_valid=config['n_data_valid']
    )
    return datamodule

def get_trainer(config, run_name):
    logger = WandbLogger(project="GRALE", 
                         name=run_name,
                         save_dir="logs",
                         tags=[])
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",  # or "auto"
        devices=1,
        max_steps=config['n_grad_steps'],
        gradient_clip_val=config['max_grad_norm'],  
        gradient_clip_algorithm="norm",
        log_every_n_steps=100,
        reload_dataloaders_every_n_epochs=1
    )
    return trainer

def main():
    # Set up
    config, run_name = get_config()
    model = get_model(config)
    datamodule = get_data(config)
    trainer = get_trainer(config, run_name)
    # Train
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()