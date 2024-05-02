#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=law_school_automl_exp1_mcar3
#SBATCH --output=%j_law_school_automl_exp1_mcar3.out
#SBATCH --error=%j_law_school_automl_exp1_mcar3.err

module purge

EXECUTION_COMMAND="
python ./scripts/impute_nulls_with_predictor.py \
    --dataset law_school \
    --null_imputers \[\"'automl'\"] \
    --run_nums \[5,6\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'exp1_mcar3'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
