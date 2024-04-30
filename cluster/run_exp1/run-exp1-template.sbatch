#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=<DATASET>_<NULL_IMPUTERS>_<EVALUATION_SCENARIO>_<MODEL>
#SBATCH --output=%j_<DATASET>_<NULL_IMPUTERS>_<EVALUATION_SCENARIO>_<MODEL>.out
#SBATCH --error=%j_<DATASET>_<NULL_IMPUTERS>_<EVALUATION_SCENARIO>_<MODEL>.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset <DATASET> \
    --null_imputers \[<NULL_IMPUTERS>\] \
    --models \[<MODEL>\] \
    --run_nums \[<RUN_NUMS>\] \
    --tune_imputers true \
    --ml_impute true \
    --evaluation_scenarios \[<EVALUATION_SCENARIO>\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
