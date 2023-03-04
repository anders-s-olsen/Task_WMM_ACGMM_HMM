#!/bin/sh
for r in 0 1 2 3 4
do

    sed -i '$ d' synthetic_noise_template.sh
    echo "python3 Synthetic_noise.py $r" >> synthetic_noise_template.sh
    bsub < synthetic_noise_template.sh

done