#!/bin/sh
#BSUB -J syntheticKjob
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -B
#BSUB -N
#BSUB -o syntheticKjob_out_%J.txt
#BSUB -e syntheticKjob_err_%J.txt
#BSUB -W 10:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load python3/3.10.7
module load numpy/1.23.3-python-3.10.7-openblas-0.3.21
module load matplotlib/3.6.0-numpy-1.23.3-python-3.10.7
module load h5py/3.7.0-python-3.10.7
python3 -m pip install --user tqdm torch
cd ..
cd notebooks
python3 Synthetic_k.py