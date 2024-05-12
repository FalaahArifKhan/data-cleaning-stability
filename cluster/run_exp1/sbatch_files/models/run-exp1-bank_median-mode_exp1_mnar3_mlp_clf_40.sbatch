#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=bank_median-mode_exp1_mnar3_mlp_clf
#SBATCH --output=%j_bank_median-mode_exp1_mnar3_mlp_clf.out
#SBATCH --error=%j_bank_median-mode_exp1_mnar3_mlp_clf.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset bank \
    --null_imputers \[\"'median-mode'\"] \
    --models \[\"'mlp_clf'\"\] \
    --run_nums \[1,2,3\] \
    --tune_imputers true \
    --ml_impute false \
    --evaluation_scenarios \[\"'exp1_mnar3'\"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
