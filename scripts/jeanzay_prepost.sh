#!/bin/bash
#SBATCH --job-name=coloring_generation          # nom du job
#SBATCH --output=logs/coloring_generation.out      # nom du fichier de sor>
#SBATCH --error=logs/coloring_generation.err       # nom du fichier d'erre>
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GP>
#SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p5 (1/8>
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la termin>
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:05:00              # temps maximum d'execution demande (HH:MM>
#SBATCH --partition=prepost
#SBATCH --account=jmk@a100

module purge # nettoyer les modules herites par defaut
module load arch/a100
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/2.3.0 # charger les modules
set -x # activer l’echo des commandes

srun python -u prepare_data.py 
