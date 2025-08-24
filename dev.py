import torch
from GRALE.model.utils import build_laplacian_node_pos, build_random_node_pos
from GRALE.data.dataset import DataModule
from GRALE.main import GRALE_model
from time import perf_counter
import yaml

# Load model
config = yaml.safe_load(open('GRALE/configs/dev.yaml', 'r'))
model = GRALE_model(config=config)
model.to('cuda')

# Load data
path_h5 = 'data/h5/PUBCHEM_dev.h5'
datamodule = DataModule(path_h5=path_h5, batch_size=16, n_data_epoch=1000, n_data_valid=100)

# Test forward pass
start_time = perf_counter()
batch_count = 0
for inputs in datamodule.train_dataloader():
    inputs = inputs.to('cuda')
    outputs = model(inputs)
    batch_count += 1
    if batch_count >= 100:
        break
end_time = perf_counter()
print(f"{(end_time - start_time)/batch_count:.2f} seconds/batch.")
