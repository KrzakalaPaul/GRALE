#!/bin/bash
#SBATCH --job-name=coloring_big_0          # nom du job
#SBATCH --output=logs/coloring_big_0%j.out      # nom du fichier de sortie
#SBATCH --error=logs/coloring_big_0%j.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -C h100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=2          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:2                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=12           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --account=jmk@h100
#SBATCH --qos_gpu_h100-dev  #decomenter pour 100h mode

module purge # nettoyer les modules herites par defaut
module load arch/h100
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/2.3.1 # charger les modules
set -x # activer l’echo des commandes
# activation du mode offline
export WANDB_MODE=offline

srun python -u train.py --run_name coloring_big_0 --dataset_path $SCRATCH.COLORING_big.h5 --config coloring_big