#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=folk_miss_forest_exp1_mnar3
#SBATCH --output=%j_folk_miss_forest_exp1_mnar3.out
#SBATCH --error=%j_folk_miss_forest_exp1_mnar3.err

module purge

EXECUTION_COMMAND="
python ../scripts/impute_nulls_with_predictor.py \
    --dataset folk \
    --null_imputers \[\"'miss_forest'\"] \
    --run_nums \[5\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'exp1_mnar3'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
