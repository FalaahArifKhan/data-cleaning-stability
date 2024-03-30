#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=rich_income_DIR_RF_5K

module purge
source $SCRATCH/fairness-variance/.venv/bin/activate

python ./scripts/evaluate_models.py \
    --dataset folk \
    --null_imputers \[\"median-mode\"\] \
    --models \[\"lr_clf\"\] \
    --run_nums \[1,2,3\] \
    --ml_impute True --evaluation_scenarios \[\"mcar_mcar1\"\]

python ./scripts/evaluate_models.py \
    --dataset folk \
    --null_imputers \[\"median-mode\"\] \
    --models \[\"lr_clf\"\] \
    --run_nums \[1\] \
    --ml_impute True --evaluation_scenarios \[\"mcar_mcar1\"\]
