#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dh3553@nyu.edu
#SBATCH --account=pr_152_general
#SBATCH --job-name=cpu_python
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module purge

EXECUTION_COMMAND="
python ../notebooks/EDA/tune_ACS_Employment.py
"

singularity exec \
	    --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro \
	    /scratch/work/public/singularity/ubuntu-20.04.1.sif \
	    /bin/bash -c "source /ext3/env.sh; ${EXECUTION_COMMAND}"
