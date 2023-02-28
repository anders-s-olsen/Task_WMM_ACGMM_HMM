#!/bin/sh
for K in 1 2 3 4 5 6 7 8 9 10
do

    sed -i '$ d' real_K_Watsontemplate.sh
    echo "python3 realdata_k.py $K" >> real_K_Watsontemplate.sh
    bsub < real_K_Watsontemplate.sh

done