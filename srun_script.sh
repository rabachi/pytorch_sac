#!/bin/bash

#SBATCH -t 59:00:00               # max runtime is 59 minutes
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --output=slurm_out/slurm_%A_%a_train.out
#SBATCH --array=0-5

module load tensorflow2-gpu-cuda10.1-conda-python3.6
module load pytorch1.7.1-cuda11.0-python3.6
# source activate pytorch_sac

echo "${SLURM_ARRAY_TASK_ID}"

# python viptrain.py env=cartpole_swingup seed=${SLURM_ARRAY_TASK_ID}

python train.py env=cartpole_swingup seed=${SLURM_ARRAY_TASK_ID}