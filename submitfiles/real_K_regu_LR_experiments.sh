#!/bin/sh
for regu in 0.000001 0.00001 0.0001 0.001 0.01 0.1
do
    for LR in 0.01 0.1 1 10
    do

        sed -i '$ d' real_K_regu_LR_template.sh
        echo "python3 realdata_regu_K_LR.py $regu $LR" >> real_K_regu_LR_template.sh
        bsub < real_K_regu_LR_template.sh

    done
done