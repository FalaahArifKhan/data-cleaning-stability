# Cluster

## Useful Materials

NYU HPC dashboard -- [https://ood-4.hpc.nyu.edu/pun/sys/dashboard](https://ood-4.hpc.nyu.edu/pun/sys/dashboard)


## Development on the Cluster

SSH to the cluster:
```shell
ssh dh3553@greene.hpc.nyu.edu
```

Get your project IDs:
```shell
sacctmgr list assoc format=user,account%25 where user=<NYU id>
```

Commands for development on the cluster using GPUs:
```shell
# To request one GPU card, 16 GB memory, and 12 hour running duration
srun -t12:00:00 --mem=16000 --gres=gpu:rtx8000:1 --pty /bin/bash

singularity exec --nv --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

Singularity> source /ext3/env.sh
```

Commands for development on the cluster using CPUs:
```shell
# To request 32 CPUs, 16 GB memory, and 12 hour running duration
srun -t12:00:00 --mem=16000 --ntasks-per-node=1 --cpus-per-task=12 --pty /bin/bash

singularity exec --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro /scratch/work/public/singularity/ubuntu-20.04.1.sif /bin/bash

Singularity> source /ext3/env.sh
```

Run jobs using a bash script:
```shell
/logs $ bash ../cluster/run_baselines/run-baselines.sh
```

Find location of a python package to change source files:
```shell
# /ext3/miniconda3/lib/python3.9/site-packages/datawig/__init__.py
python -c "import datawig; print(datawig.__file__)"
```

SSH to a bash SLURM job:
```shell
# Pattern: ssh username@hostname
ssh dh3553@cs223
```

Count the number of files in each directory:
```shell
du -a | cut -d/ -f2 | sort | uniq -c | sort -nr
```


## Setup

Configurate dependencies on the cluster ([ref](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)):

```shell
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz

singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
bash Miniconda3-py39_24.1.2-0-Linux-x86_64.sh -b -p /ext3/miniconda3
# rm Miniconda3-py39_24.1.2-0-Linux-x86_64.sh # if you don't need this file any longer

pip3 install -r requirements.txt
pip3 install mxnet-cu110
pip3 install datawig --no-deps

# https://stackoverflow.com/questions/54249577/importerror-libcuda-so-1-cannot-open-shared-object-file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/compat

singularity exec --overlay /scratch/dh3553/ml_life_cycle_project/vldb_env.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c 'source /ext3/env.sh; python -c "import datawig; print(datawig.__file__)"'

singularity exec /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
