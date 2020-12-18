#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=08:00:00,ncpus=1,mem=20GB,jobfs=1GB,software=python
#PBS -l storage=gdata/q27
#PBS -l wd

module load python3/3.8.5

python3 genBNPAL.py >> genBNPAL_RCS.log
