#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=diabetes_median-dummy_exp2_3_mcar1_rf_clf
#SBATCH --output=%j_diabetes_median-dummy_exp2_3_mcar1_rf_clf.out
#SBATCH --error=%j_diabetes_median-dummy_exp2_3_mcar1_rf_clf.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset diabetes \
    --null_imputers \[\"'median-dummy'\"] \
    --models \[\"'rf_clf'\"\] \
    --run_nums \[1,2,3,4,5,6\] \
    --tune_imputers true \
    --ml_impute false \
    --evaluation_scenarios \[\"'exp2_3_mcar1'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
