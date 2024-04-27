#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=heart_lgbm_clf
#SBATCH --output=%j_heart_lgbm_clf.out
#SBATCH --error=%j_heart_lgbm_clf.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_baseline.py \
    --dataset heart \
    --models \[\"'lgbm_clf'\"\] \
    --run_nums \[3,4\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
