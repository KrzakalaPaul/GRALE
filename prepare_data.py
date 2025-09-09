from datasets.molecules.PUBCHEM import download_PUBCHEM, process_PUBCHEM
from datasets.coloring.data_generation import create_coloring_dataset
import json
import os 

def PUBCHEM():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/h5", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # Download datasets
    download_PUBCHEM()

    # Process a small version for development
    config = json.load(open("datasets/molecules/configs/dev.json"))
    process_PUBCHEM(config, dev=True)
    
    # Process a version with only molecules up to 16 atoms
    config = json.load(open("datasets/molecules/configs/16.json"))
    process_PUBCHEM(config, dev=False)
    
def COLORING():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/h5", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    
    config = json.load(open('datasets/coloring/configs/small.json'))
    create_coloring_dataset(config)
    
    config = json.load(open('datasets/coloring/configs/medium.json'))
    create_coloring_dataset(config)
    

if __name__ == "__main__":
    COLORING()