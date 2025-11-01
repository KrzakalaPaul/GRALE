from GRALE.data.dataset import DataModule
from GRALE.main import GRALE_model
import yaml
import torch
from ot import emd
import numpy as np
from scipy.optimize import linear_sum_assignment

def my_emd(M):
    n = len(M)
    _, permutation = linear_sum_assignment(M.T)
    T = np.eye(n)[:,permutation]/n
    return T

M = (1 - np.eye(3)) + 0.1 * np.random.rand(3,3)
out = emd(M=M, a=[], b=[], numItermax=10, log=True)
out2 = my_emd(M)
print(out[0])
print(out2)

if __name__ == "__main__" and False:
    torch.set_printoptions(precision=1, sci_mode=False)
    path_h5 = 'data/h5/COLORING_medium.h5'
    n_gpus = 1
    config = yaml.safe_load(open('GRALE/configs/coloring_medium.yaml'))
    datamodule = DataModule(
        path_h5=path_h5,
        batch_size=int(config['batchsize_effective']/n_gpus),
        n_data_epoch=config['n_data_epoch'],
        n_data_valid=config['n_data_valid'],
        n_workers=config['n_workers']
    )
    model = GRALE_model(config=config)
    batch = next(iter(datamodule.train_dataloader()))
    permutation_list = model.canonical_permutation(batch, hard_matcher=True)
    print(permutation_list[0])