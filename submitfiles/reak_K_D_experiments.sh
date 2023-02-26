#!/bin/sh
for K in 1 2 3 4 5 6 7 8 9 10
do

    sed -i '$ d' real_K_D_template.sh
    echo "python3 realdata_K_D.py $K" >> real_K_D_template.sh
    bsub < real_K_D_template.sh

done