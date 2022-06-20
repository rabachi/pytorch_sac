#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00      # time limit
#SBATCH --mem=32GB         # minimum amount of real memory
#SBATCH --job-name=viper

source ~/.bashrc
export MUJOCO_PY_BYPASS_LOCK=True
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/voelcker/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

module load cuda-11.3
pyenv shell bayesian_daml

python -m sac.main env_name=$1 num_ensemble=$2 seed=$RANDOM hydra.run.dir=/checkpoint/voelcker/$SLURM_JOB_ID

