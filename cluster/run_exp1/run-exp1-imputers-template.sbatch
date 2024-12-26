#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>
#SBATCH --output=%j_<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>.out
#SBATCH --error=%j_<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>.err

module purge

EXECUTION_COMMAND="
python ../scripts/impute_nulls_with_predictor.py \
    --dataset <DATASET> \
    --null_imputers \[\"'<NULL_IMPUTER>'\"] \
    --run_nums \[<RUN_NUMS>\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'<EVALUATION_SCENARIO>'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
