#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=10:00:00,ncpus=1,mem=5GB,jobfs=1GB,software=python
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

module load python3/3.8.5

python3 AuPtgenDAPfilesL10L12.py 
