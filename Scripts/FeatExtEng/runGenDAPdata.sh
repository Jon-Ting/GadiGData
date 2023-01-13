#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=48,walltime=24:00:00,mem=50GB,jobfs=0GB
#PBS -l storage=gdata/q27+scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.11.0

python3 genCSVs.py  #PBS -l ncpus=1,walltime=02:00:00,mem=50GB,jobfs=0GB  #PBS -l ncpus=48,walltime=10:00:00,mem=30GB,jobfs=0GB
#python3 mergeFeatures.py #PBS -l ncpus=48,walltime=01:00:00,mem=36GB,jobfs=0GB  #PBS -l ncpus=1,walltime=00:20:00,mem=25GB,jobfs=0GB
