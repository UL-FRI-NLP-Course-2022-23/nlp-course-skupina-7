#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem 10G
#SBATCH --time=20:00:00
#SBATCH --output=logs/translation-%J.out
#SBATCH --error=logs/translation-%J.err
#SBATCH --job-name="Translation Job"

srun python translate.py
