#!/bin/sh
#BSUB -J syntheticjob
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -B
#BSUB -N
#BSUB -o syntheticjob_out_%J.txt
#BSUB -e syntheticjob_err_%J.txt
#BSUB -W 10:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load python3/3.10.7
module load numpy
module load matplotlib
module load h5py
python -m pip3 install --user tqdm torch
cd ..
cd notebooks
python Synthetic.py
matlab -nodisplay -batch analyze_k_results