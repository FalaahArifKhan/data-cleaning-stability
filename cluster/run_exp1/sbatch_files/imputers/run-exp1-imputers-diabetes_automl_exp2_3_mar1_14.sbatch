#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=diabetes_automl_exp2_3_mar1
#SBATCH --output=%j_diabetes_automl_exp2_3_mar1.out
#SBATCH --error=%j_diabetes_automl_exp2_3_mar1.err

module purge

EXECUTION_COMMAND="
python ../scripts/impute_nulls_with_predictor.py \
    --dataset diabetes \
    --null_imputers \[\"'automl'\"] \
    --run_nums \[4,5,6\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'exp2_3_mar1'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
