#!/bin/sh
#SBATCH --job-name=finetune_nlp_paraphraser # job name
#SBATCH --output=train_output_sbatch.txt # file name to redirect the output
#SBATCH --time=00:40:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --gpus=1 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --mem-per-gpu=64G # memory allocation
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12 # number of allocated cores

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate nlp_mt5_paraphrasing # activate the previously created environment
srun --nodes=1 --exclusive --gpus=1 --ntasks=1 python nlp_paraphraser_TEST.py
