#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=diabetes_datawig_exp1_mnar3
#SBATCH --output=%j_diabetes_datawig_exp1_mnar3.out
#SBATCH --error=%j_diabetes_datawig_exp1_mnar3.err

module purge

EXECUTION_COMMAND="
python ../scripts/impute_nulls_with_predictor.py \
    --dataset diabetes \
    --null_imputers \[\"'datawig'\"] \
    --run_nums \[1,2,3\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'exp1_mnar3'\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
