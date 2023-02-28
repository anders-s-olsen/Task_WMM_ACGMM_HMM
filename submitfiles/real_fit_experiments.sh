#!/bin/sh
for m in 0 1
do
for r in 0 1 2 3 4
do

    sed -i '$ d' real_fit_template.sh
    echo "python3 realdata_fit.py $m $r" >> real_fit_template.sh
    bsub < real_fit_template.sh

done
# done