#!/bin/bash -x
#SBATCH --account=gsp19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=BBB-out.%j
#SBATCH --error=BBB-err.%j
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpus

srun python Bayesian_CNN_Detailed.py
