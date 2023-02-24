#!/bin/sh

for r in {1..5}
do
for K in {1..10}


sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 1' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 2' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 3' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 4' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 5' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 6' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 7' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 8' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 9' >> realkr_template.sh
bsub < realkr_template.sh

sed '$d' realkr_template.sh
echo 'python3 realdata_K_LS_splithalf.py 10' >> realkr_template.sh
bsub < realkr_template.sh