#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=01:00:00,ncpus=1,mem=20GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to append all *.lmp files back into the corresponding S0 tar ball
# Author: Jonathan Yik Chang Ting
# Date: 10/1/2021

declare -a TYPE_ARR=('L10/')
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    annealDirName=$SIM_DATA_DIR/SimAnneal/$bnpType
    for bnpDir in $annealDirName*/;
        do count=`ls -1 ${bnpDir}*.lmp 2>/dev/null | wc -l`
        if [ $count != 0 ]; then 
            echo ${bnpDir::-1} $count
            cd ${bnpDir}; tar -xf ${bnpDir::-1}S0.tar.gz
            mv *lmp ${bnpDir::-1}S0
            tar -czf ${bnpDir::-1}S0.tar.gz ${bnpDir::-1}S0
            rm -rf ${bnpDir::-1}S0
            cd ..
        fi
    done
done
# Yet to be tested!
