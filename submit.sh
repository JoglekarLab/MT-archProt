#!/bin/bash
#SBATCH --job-name=mt_oligo
#SBATCH --account=ajitj99
#SBATCH --partition=standard
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
 
eval "$(conda shell.bash hook)"
conda activate /nfs/turbo/umms-ajitj/conda_envs/myenv

python simulationv2.py