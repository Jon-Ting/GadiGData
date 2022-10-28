#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=100GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l other=mdss
#PBS -l wd


mdss get $USER/SimAnneal/simTraj/L10All.zip .
mdss get $USER/SimAnneal/simTraj/L12All.zip .
