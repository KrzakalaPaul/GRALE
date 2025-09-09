#!/bin/sh

#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

set -x

srun python -u train.py --run_name coloring_more_layers --config coloring_more_layers