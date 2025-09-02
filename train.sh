#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=L40S
#SBATCH --gpus=2

set -x

srun python -u train.py --run_name PUBCHEM16_1 --config 16