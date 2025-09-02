#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=40:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name PUBCHEM16_softsort --config 16_softsort