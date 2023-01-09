#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=5GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to move backed-up files from massdata/ to gdata/
# Author: Jonathan Yik Chang Ting
# Date: 6/5/2022

for tarredNP in $(mdss ls $USER/SimAnneal/simTraj/CS60+); do
    if [ -f "CSAu60+/$tarredNP" ]; then continue; fi 
    NPname=$(echo $tarredNP)
    if [[ "$NPname" != 'Au'* ]]; then continue; fi
    echo $NPname
    mdss get $USER/SimAnneal/simTraj/CS60+/$NPname CSAu60+/ 
done
