for regu in 0.000001,0.00001,0.0001,0.001,0.01,0.1
do
for LR in 0.01,0.1,1,10
do
bsub -J bigexp -o bigexp_%J.out -e bigexp_%J.err -q hpc -n 8 -R "rusage[mem=2G]" -R "span[hosts=1]" -W 10:00 -B -N > module load python3/3.10.7 && \
module load numpy/1.23.3-python-3.10.7-openblas-0.3.21 && \
module load matplotlib/3.6.0-numpy-1.23.3-python-3.10.7 && \
module load h5py/3.7.0-python-3.10.7 && \
python3 -m pip install --user tqdm torch && \
cd .. && \
cd notebooks && \
python3 realdata_regu_K_LR.py $regu $LR
done
done