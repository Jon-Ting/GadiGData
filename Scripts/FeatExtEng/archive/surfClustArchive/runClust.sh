#!/bin/bash
#PBS -P p00
#PBS -q normal
#PBS -l ncpus=24,walltime=02:30:00,mem=120GB,jobfs=1GB
#PBS -l storage=scratch/p00
#PBS -l wd
#PBS -m a

module load python3/3.8.5

python3 surfClust.py AuPt20COL12 0 2501 ILS
