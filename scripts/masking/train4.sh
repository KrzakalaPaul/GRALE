#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name masking_20 --config masking/16_masking_20 --dataset PUBCHEM_16