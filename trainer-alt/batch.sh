#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/paraphrasing-%J.out
#SBATCH --error=logs/paraphrasing-%J.err
#SBATCH --job-name="Paraphrasing model training"

export TOKENIZERS_PARALLELISM=false
srun python mt5.py
