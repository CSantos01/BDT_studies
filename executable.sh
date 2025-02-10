set -e
cd /afs/desy.de/user/c/csantos/BDT_studies/
echo 'Working in the folder:'; pwd
echo 'Current environment:'; env
echo 'Will now execute the program'
exec python3 metrics_testing.py --n_sample 1000000 --weights 0.98 0.02 --extra_label 1M