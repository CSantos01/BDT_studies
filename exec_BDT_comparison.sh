set -e
cd /afs/desy.de/user/c/csantos/BDT_studies/
echo 'Working in the folder:'; pwd
echo 'Current environment:'; env
echo 'Will now execute the program'
exec python3 BDT_comparison.py --n_sample 1000000 --weights 0.99 0.01 --extra_label 1M_99_01