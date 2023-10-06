#!/bin/bash

#SBATCH --job-name=train_1d_benchmarks
#SBATCH --output=train_1d_benchmarks.log
#SBATCH --error=train_1d_benchmarks.err
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

T_list=(50 100)
for tmax in ${T_list[@]}
do
    python train.py -T $tmax
done