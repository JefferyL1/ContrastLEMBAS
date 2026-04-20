#!/bin/bash
#SBATCH --job-name=lim_contrastive         	# Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=4             # CPU cores per task
#SBATCH --mem=16G                      # Memory
#SBATCH --gpus=1                      # Number of GPUs
#SBATCH --time=06:00:00               # Walltime (HH:MM:SS)
#SBATCH --partition=mit_normal_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jefferyl@mit.edu


# Activate your Conda environment
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/contrast_lembas/lib:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrast_lembas

# Create folder and redirect stdoutt and stderr
JOB_FOLDER="/orcd/home/002/jefferyl/orcd/scratch/ContrastLEMBAS/contrastive_model/results/linearNN_lim"
mkdir -p $JOB_FOLDER

# Run python
cd /orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/code

python3 model/run_model.py $JOB_FOLDER --limited
       	
