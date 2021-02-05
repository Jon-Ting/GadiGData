#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=40:00:00,ncpus=1,mem=20GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

# Goal: Script to finish the post execution tasks of unfinished jobs
# Author: Jonathan Yik Chang Ting
# Date: 10/1/2021

SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
for tarFile in $SIM_DATA_DIR/*/*/*S0.tar.gz; do
    isErr=$(tar -tzf $tarFile | head -n 1 | grep "scratch")
    if [ ! -z $isErr ]; then
        jobName=/${isErr::-1}; dirName=${jobName%/*}; unqName=$(echo $jobName | awk -F'/' '{print $NF}')
        echo $dirName >> fixTarErr.log; cd $dirName
        tar -xf $tarFile; mkdir $unqName; mv scratch/*/*/*/*/*/*/*lmp $unqName
        tar -czf $unqName.tar.gz $unqName; rm -rf $unqName scratch
    fi
done
