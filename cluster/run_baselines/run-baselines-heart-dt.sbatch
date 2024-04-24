#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --output=%j_heart_dt.out
#SBATCH --error=%j_heart_dt.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_baseline.py \
    --dataset heart \
    --models \[\"'dt_clf'\"\] \
    --run_nums \[1,2,3,4,5,6\]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
