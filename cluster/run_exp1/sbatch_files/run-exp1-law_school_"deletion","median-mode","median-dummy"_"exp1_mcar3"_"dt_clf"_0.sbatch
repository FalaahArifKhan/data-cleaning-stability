#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --job-name=law_school_"deletion","median-mode","median-dummy"_"exp1_mcar3"_"dt_clf"
#SBATCH --output=%j_law_school_"deletion","median-mode","median-dummy"_"exp1_mcar3"_"dt_clf".out
#SBATCH --error=%j_law_school_"deletion","median-mode","median-dummy"_"exp1_mcar3"_"dt_clf".err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_models.py \
    --dataset law_school \
    --null_imputers \["deletion","median-mode","median-dummy"\] \
    --models \["dt_clf"\] \
    --run_nums \[1,2,3,4,5,6\] \
    --tune_imputers true \
    --ml_impute true \
    --evaluation_scenarios \["exp1_mcar3"\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
