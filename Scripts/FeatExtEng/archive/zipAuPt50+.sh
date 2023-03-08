#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=50GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

eleComb=AuPt
for i in /scratch/q27/jt5911/$eleComb/${eleComb}50+/*; do
    zip -6 -ru /scratch/$PROJECT/$USER/$eleComb/${eleComb}50+.zip $i
done
