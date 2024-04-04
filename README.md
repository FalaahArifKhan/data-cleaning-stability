# data-cleaning-stability

Studying the impact of data cleaning techniques on fairness and stability


## Setup

Install datawig:
```shell
pip install mxnet-cu110
pip install datawig --no-deps
```

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
