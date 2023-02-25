for regu in 0.000001
do
for LR in 0.01
do
bsub -J bigexp -o bigexp_%J.out -e bigexp_%J.err -q hpc -n 8 -R "rusage[mem=2G]" -R "span[hosts=1]" -W 10:00 -B -N > "module load python3/3.10.7 \n module load numpy/1.23.3-python-3.10.7-openblas-0.3.21 \n module load matplotlib/3.6.0-numpy-1.23.3-python-3.10.7 \n module load h5py/3.7.0-python-3.10.7 \n python3 -m pip install --user tqdm torch \n cd .. \n cd notebooks \n python3 realdata_regu_K_LR.py $regu $LR"
done
done