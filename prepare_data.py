from datasets.molecules.PUBCHEM import download_PUBCHEM, process_PUBCHEM
import json

def main():
    # Download datasets
    download_PUBCHEM()

    # Load configuration
    config = json.load(open("datasets/molecules/configs/dev.json"))
    
    process_PUBCHEM(config, dev=True)
    

if __name__ == "__main__":
    main()