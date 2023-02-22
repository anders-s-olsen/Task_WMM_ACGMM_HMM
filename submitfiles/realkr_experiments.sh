#!/bin/sh
sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 1' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 2' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 3' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 4' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 5' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 6' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 7' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 8' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 9' >> realk_template.sh
bsub < realk_template.sh

sed '$d' realk_template.sh
echo 'python3 realdata_K_LS_splithalf.py 10' >> realk_template.sh
bsub < realk_template.sh