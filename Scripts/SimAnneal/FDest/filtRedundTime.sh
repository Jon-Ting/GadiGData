#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB,software=python
#PBS -l storage=gdata/q27
#PBS -l wd

module load python3/3.7.4

python3 filtRedund.py > AuCo50RDCS_1957.log
