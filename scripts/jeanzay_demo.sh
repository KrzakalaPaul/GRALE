#!/bin/bash
#SBATCH --job-name=job          # nom du job
#SBATCH --output=logs/gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=logs/gpu_mono%j.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:2                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=16           # nombre de CPU par tache pour gpu_p5 
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:10:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --account=jmk@a100
#SBATCH --qos=qos_gpu_a100-dev # decomenter pour "dev" mode

module purge # nettoyer les modules herites par defaut
module load arch/a100
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/2.3.0 # charger les modules
set -x # activer l’echo des commandes
# activation du mode offline
export WANDB_MODE=offline

srun python -u train.py --run_name demo --dataset PUBCHEM_32 --config 32