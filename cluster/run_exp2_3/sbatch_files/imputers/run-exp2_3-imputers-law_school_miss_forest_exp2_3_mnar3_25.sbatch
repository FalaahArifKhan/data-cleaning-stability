#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=96GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=law_school_miss_forest_exp2_3_mnar3
#SBATCH --output=%j_law_school_miss_forest_exp2_3_mnar3.out
#SBATCH --error=%j_law_school_miss_forest_exp2_3_mnar3.err

module purge

EXECUTION_COMMAND="
python ../scripts/impute_nulls_with_predictor.py \
    --dataset law_school \
    --null_imputers \[\"'miss_forest'\"] \
    --run_nums \[3,4\] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios \[\"'exp2_3_mnar3'\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
