from datasets.coloring.data_generation import create_coloring_dataset
import json

if __name__ == "__main__":
    config = json.load(open('datasets/coloring/configs/medium.json'))
    create_coloring_dataset(config)