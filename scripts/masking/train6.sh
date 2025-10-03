#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=V100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name masking_10_larger --config masking/16_masking_10_larger --dataset PUBCHEM_16