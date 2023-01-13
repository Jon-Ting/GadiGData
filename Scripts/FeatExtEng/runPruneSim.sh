#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=20GB,jobfs=1GB
#PBS -l storage=gdata/q27+scratch/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to run pruneSimFiles.py
# Author: Jonathan Yik Chang Ting
# Date: 6/5/2022

module load python3/3.11.0

python3 ./pruneSimFiles.py
