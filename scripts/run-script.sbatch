#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --output=datawig_%j.out
#SBATCH --error=datawig_%j.err

module purge

EXECUTION_COMMAND="
python ./scripts/impute_nulls_with_predictor.py \
    --dataset folk \
    --null_imputers \[\"datawig\"\] \
    --run_nums \[1\] \
    --save_imputed_datasets True --evaluation_scenarios \[\"mcar_mcar1\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
