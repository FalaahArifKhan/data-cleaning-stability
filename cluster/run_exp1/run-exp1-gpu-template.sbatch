#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=<DATASET>_datawig_<EVALUATION_SCENARIO>_<MODEL>
#SBATCH --output=%j_<DATASET>_datawig_<EVALUATION_SCENARIO>_<MODEL>.out
#SBATCH --error=%j_<DATASET>_datawig_<EVALUATION_SCENARIO>_<MODEL>.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset <DATASET> \
    --null_imputers \[\"'datawig'\"] \
    --models \[\"'<MODEL>'\"\] \
    --run_nums \[<RUN_NUMS>\] \
    --tune_imputers true \
    --ml_impute true \
    --evaluation_scenarios \[\"'<EVALUATION_SCENARIO>'\"\]
"

singularity exec --nv \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
