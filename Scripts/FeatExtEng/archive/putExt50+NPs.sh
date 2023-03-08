#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=40GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l other=mdss
#PBS -l wd

echo 'CSPd50+'
#mdss put /scratch/q27/jt5911/BNP_MDsim/RAL50+/RALPd50+.zip $USER/SimAnneal/finalStructProp/
#mdss put /scratch/q27/jt5911/BNP_MDsim/RAL50+/RALPt50+.zip $USER/SimAnneal/finalStructProp/
#mdss put /scratch/q27/jt5911/BNP_MDsim/RCS50+/RCSPt50+.zip $USER/SimAnneal/finalStructProp/
#mdss put /scratch/q27/jt5911/BNP_MDsim/RCS50+/RCSCo50+.zip $USER/SimAnneal/finalStructProp/
mdss put /scratch/q27/jt5911/BNP_MDsim/CS50+/CSPd50+.zip $USER/SimAnneal/finalStructProp/
