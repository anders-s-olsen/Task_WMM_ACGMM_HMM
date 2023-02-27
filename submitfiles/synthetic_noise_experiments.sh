#!/bin/sh
for r in 0 1 2 3 4
do

    sed -i '$ d' submit_synthetic_noise.sh
    echo "python3 Synthetic_noise.py $r" >> submit_synthetic_noise.sh
    bsub < submit_synthetic_noise.sh

done