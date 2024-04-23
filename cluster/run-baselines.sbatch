#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --output=datawig_%j.out
#SBATCH --error=datawig_%j.err

module purge

EXECUTION_COMMAND="
python ../scripts/evaluate_baseline.py \
    --dataset folk \
    --models [\"lr_clf\"] \
    --run_nums [1]
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
