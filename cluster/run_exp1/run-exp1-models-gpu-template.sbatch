#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=96GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>
#SBATCH --output=%j_<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>.out
#SBATCH --error=%j_<DATASET>_<NULL_IMPUTER>_<EVALUATION_SCENARIO>.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset <DATASET> \
    --null_imputers \[\"'<NULL_IMPUTER>'\"] \
    --models \[\"'<MODEL>'\"\] \
    --run_nums \[<RUN_NUMS>\] \
    --tune_imputers true \
    --ml_impute false \
    --evaluation_scenarios \[\"'<EVALUATION_SCENARIO>'\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
