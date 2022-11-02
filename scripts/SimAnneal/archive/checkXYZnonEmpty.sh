#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

eleComb=AuPd
for confID in /scratch/$PROJECT/$USER/SimAnneal/$eleComb/00000*/*xyz; do
    echo $confID
    if [[ -z $(grep [^[:space:]] $confID) ]]; then 
        echo "    Empty!"
    fi
done
