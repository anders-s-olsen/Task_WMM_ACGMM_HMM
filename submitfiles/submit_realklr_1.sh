#!/bin/sh
#BSUB -J realKjob
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -B
#BSUB -N
#BSUB -o realKjob_out_%J.txt
#BSUB -e realKjob_err_%J.txt
#BSUB -W 72:00 
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
python3 realdata_K_LS_splithalf.py 1