#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=48,walltime=24:00:00,mem=128GB,jobfs=0GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.11.0

python3 genDAPfiles.py > runGenDAPfiles.log  #PBS -l ncpus=48,walltime=05:00:00,mem=128GB,jobfs=0GB
python3 mergeFeatures.py #PBS -l ncpus=1,walltime=00:30:00,mem=20GB,jobfs=0GB
