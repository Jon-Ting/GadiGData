#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to move backed-up files from massdata/ to gdata/
# Ptthor: Jonathan Yik Chang Ting
# Date: 6/5/2022

element='Co'
for tarredNP in $(mdss ls $USER/SimAnneal/simTraj/CS60+); do
    NPname=$(echo $tarredNP)
    if [[ "$NPname" != $element* ]]; then continue; fi
    if [ -f "/scratch/$PROJECT/$USER/BNP_MDsim/CS50+/CS${element}50+/$tarredNP" ]; then continue; fi 
    if [ -f "/scratch/$PROJECT/$USER/BNP_MDsim/CS50+/CS${element}50+/${tarredNP:0:-7}/${tarredNP:0:-7}S2.log" ]; then 
        if [ -f "/scratch/$PROJECT/$USER/BNP_MDsim/CS50+/CS${element}50+/${tarredNP:0:-7}/${tarredNP:0:-7}S2.zip" ]; then continue; fi
    fi 
    echo "Getting $NPname..."
    mdss get $USER/SimAnneal/simTraj/CS60+/$NPname /scratch/$PROJECT/$USER/BNP_MDsim/CS50+/CS${element}50+/ 
    rm -rf /scratch/$PROJECT/$USER/BNP_MDsim/CS50+/CS${element}50+/${tarredNP:0:-7}/
done
echo $element
