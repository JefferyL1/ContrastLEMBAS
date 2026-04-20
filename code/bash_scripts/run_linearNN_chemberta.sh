#!/bin/bash
#SBATCH --job-name=chemberta         	# Job name
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
JOB_FOLDER="/orcd/home/002/jefferyl/orcd/scratch/ContrastLEMBAS/contrastive_model/results/linearNN_chemberta"
mkdir -p $JOB_FOLDER

# Run python (full)
cd /orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/code
DRUG_EMBEDDINGS="/home/jefferyl/ContrastLEMBAS/contrastive_model/data/drug_data/embeddings/smiles_to_chemberta.pkl"
python3 model/run_model.py --output_dir $JOB_FOLDER --drug_embeddings $DRUG_EMBEDDINGS

# Create folder and redirect stdoutt and stderr
JOB_FOLDER_LIM="/orcd/home/002/jefferyl/orcd/scratch/ContrastLEMBAS/contrastive_model/results/linearNN_chemberta_lim"
mkdir -p $JOB_FOLDER_LIM

# Run python (lim)
cd /orcd/home/002/jefferyl/ContrastLEMBAS/contrastive_model/code
DRUG_EMBEDDINGS="/home/jefferyl/ContrastLEMBAS/contrastive_model/data/drug_data/embeddings/smiles_to_chemberta.pkl"
python3 model/run_model.py --output_dir $JOB_FOLDER_LIM --drug_embeddings $DRUG_EMBEDDINGS --limited
