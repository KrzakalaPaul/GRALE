#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name masking_5 --config masking/16_masking_5 --dataset PUBCHEM_16