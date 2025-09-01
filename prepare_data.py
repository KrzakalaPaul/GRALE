from datasets.molecules.PUBCHEM import download_PUBCHEM, process_PUBCHEM
import json
import os 
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/h5", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Download datasets
    download_PUBCHEM()

    config = json.load(open("datasets/molecules/configs/dev.json"))
    process_PUBCHEM(config, dev=True)
    
    config = json.load(open("datasets/molecules/configs/16.json"))
    process_PUBCHEM(config, dev=False)
    
    

if __name__ == "__main__":
    main()