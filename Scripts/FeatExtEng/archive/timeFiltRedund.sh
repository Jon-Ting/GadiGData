#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=01:00:00,ncpus=1,mem=5GB,jobfs=0GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

module load python3/3.11.0

python3 filtRedund.py
