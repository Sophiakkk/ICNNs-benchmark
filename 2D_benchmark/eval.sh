#!/bin/bash

#SBATCH --job-name=eval_2d_benchmarks
#SBATCH --output=eval_2d_benchmarks.log
#SBATCH --error=eval_2d_benchmarks.err
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

python evaluation.py >> output.txt
