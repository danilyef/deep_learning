#!/bin/bash
#SBATCH --job-name=gpt_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/dan_yef/deep_learning/deep_learning/gpt/training
#SBATCH --output=slurm-%j.out

nvidia-smi
/home/dan_yef/deep_learning/venv/bin/python -u gpt_train.py
