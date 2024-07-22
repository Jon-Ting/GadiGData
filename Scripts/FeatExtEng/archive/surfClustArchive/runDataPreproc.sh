#!/bin/bash
#PBS -P p00
#PBS -q normal
#PBS -l ncpus=24,walltime=00:30:00,mem=120GB,jobfs=60GB
#PBS -l storage=scratch/p00
#PBS -l wd
#PBS -m a

module load python3/3.8.5

python3 surfDataPreproc150.py AuPt20COL12 0 2501
