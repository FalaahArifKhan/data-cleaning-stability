#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=law_school_automl_exp1_mnar3_rf_clf
#SBATCH --output=%j_law_school_automl_exp1_mnar3_rf_clf.out
#SBATCH --error=%j_law_school_automl_exp1_mnar3_rf_clf.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset law_school \
    --null_imputers \[\"'automl'\"] \
    --models \[\"'rf_clf'\"\] \
    --run_nums \[1,2,3,4,5,6\] \
    --tune_imputers true \
    --ml_impute true \
    --evaluation_scenarios \[\"'exp1_mnar3'\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
