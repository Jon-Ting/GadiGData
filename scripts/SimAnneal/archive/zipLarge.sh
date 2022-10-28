#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=150GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l wd


#zip -6 -li -r L10large.zip L10large/Pt* -u
mdss put largeL12.zip jt5911/SimAnneal/
mdss put largeL10.zip jt5911/SimAnneal/
