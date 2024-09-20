#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=<DATASET>_<MODEL>
#SBATCH --output=%j_<DATASET>_<MODEL>.out
#SBATCH --error=%j_<DATASET>_<MODEL>.err

module purge

EXECUTION_COMMAND="
python ../notebooks/EDA/tune_ACS_Employment.py
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:rw \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
