#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=150GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to tar up all S0 lmp files in a given BNP directory
# Author: Jonathan Yik Chang Ting
# Date: 1/8/2021

for dir in Pt*; do
    echo $dir
    if [ ! -d $dir ]; then continue; fi
    cd $dir
    if [ -e run.lock ]; then continue; fi
    #rm -Rf -- */
    if ls ${dir}S0.*.lmp 1> /dev/null 2>&1; then 
        echo "$dir"
        tar -xf ${dir}S0.tar.gz
        mv ${dir}S0.*.lmp ${dir}S0
        tar -czf ${dir}S0.tar.gz ${dir}S0 && rm -rf ${dir}S0
    fi
    cd ..
done
