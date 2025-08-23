from GRALE.utils.dataset import LazyDataset, InMemoryDataset, MyDataModule
import h5py

path_h5 = 'data/h5/PUBCHEM_dev.h5'
datamodule = MyDataModule(path_h5=path_h5, batch_size=32, n_data_epoch=-1, n_data_valid=500)
dataloader = datamodule.val_dataloader()
for batch in dataloader:
    print(batch.h.sum(-1))
    break