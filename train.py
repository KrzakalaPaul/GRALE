from GRALE.data.dataset import DataModule
from GRALE.main import GRALE_model
import argparse
import os
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='32')
    parser.add_argument('--run_name', type=str, default='32_test')
    parser.add_argument('--dataset', type=str, default='PUBCHEM_32')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    run_name = args.run_name
    dataset = args.dataset
    if checkpoint_path is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        config = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)["hyper_parameters"]
    else:
        config = yaml.safe_load(open(f'GRALE/configs/{args.config}.yaml', 'r'))
    return config, run_name, dataset, checkpoint_path

def get_model(config):
    model = GRALE_model(config=config)
    return model

def get_data(config, dataset):
    path_h5 = f'data/h5/{dataset}.h5'
    n_gpus = get_num_gpus()
    datamodule = DataModule(
        path_h5=path_h5,
        batch_size=int(config['batchsize_effective']/n_gpus),
        n_data_epoch=config['n_data_epoch'],
        n_data_valid=config['n_data_valid'],
        n_workers=config['n_workers']
    )
    return datamodule

def get_num_gpus():
    # USE
    if "SLURM_GPUS_ON_NODE" in os.environ:
        return int(os.environ.get("SLURM_GPUS_ON_NODE"))
    # Fallback: SLURM var
    return 1

def get_trainer(config, run_name):
    # Get Logger
    logger = WandbLogger(project="GRALE", 
                         name=run_name,
                         save_dir="logs",
                         tags=[])
    # Define checkpoint callback (with no conflict with logger)
    # Explicit checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",  # no conflict with logger
        filename="{epoch}-{edit_graph:.4f}",
        save_last=True,
        save_top_k=1,
        monitor="edit_graph",
        mode="min",
    )
    
    n_gpus = get_num_gpus()
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
        callbacks=[checkpoint_cb],
        precision=config["precision"]
    )
    return trainer

def main():
    # Set up
    config, run_name, dataset, checkpoint_path = get_config()
    model = get_model(config)
    datamodule = get_data(config, dataset)
    trainer = get_trainer(config, run_name)
    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    print()
    
if __name__ == "__main__":
    main()