#!/bin/sh
for K in 1 4 7 10
do

    sed -i '$ d' real_K4D1_template.sh
    echo "python3 realdata_ACGK4D1_initexperiment.py $K" >> real_K4D1_template.sh
    bsub < real_K4D1_template.sh

done