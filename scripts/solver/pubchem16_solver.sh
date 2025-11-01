#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=H100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name pubchem16_solver --config 16 --dataset_path data/h5/PUBCHEM_16.h5

