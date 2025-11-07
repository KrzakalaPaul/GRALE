#!/bin/bash
#SBATCH --job-name=coloring_generation          # nom du job
#SBATCH --output=logs/coloring_generation.out      # nom du fichier de sortie
#SBATCH --error=logs/coloring_generation.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=01:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --partition=prepost

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/2.3.1 # charger les modules
set -x # activer l’echo des commandes

srun python -u prepare_data.py 