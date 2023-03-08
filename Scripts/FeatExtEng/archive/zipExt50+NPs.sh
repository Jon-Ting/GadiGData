#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=30GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

declare -a TYPE_ARR=('CS')
declare -a ELE_ARR=('Co')
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    for ((j=0;j<${#ELE_ARR[@]};j++)); do
        eleType=${ELE_ARR[$j]}
        zip -6 -ru /scratch/$PROJECT/$USER/BNP_MDsim/${bnpType}50+/$bnpType${eleType}50+.zip /scratch/$PROJECT/$USER/BNP_MDsim/${bnpType}50+/$bnpType${eleType}50+/
    done
done
