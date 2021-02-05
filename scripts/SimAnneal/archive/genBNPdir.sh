#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=05:00:00,ncpus=1,mem=8GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

# Goal: Automate the generation of simulation directory for each new bimetallic nanoparticle created
# Author: Jonathan Yik Chang Ting
# Date: 5/2/2021

declare -a TYPE_ARR=('L10' 'L12' 'CS' 'RCS' 'RAL')
declare -a TYPE_ARR=('CS')  # DEBUG
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
INIT_STRUCT_DIR=/g/data/$PROJECT/$USER/InitStruct/BNP
CONFIG_FILE=config.yml

echo "Looping through directories:"
echo "-----------------------------------------------"
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    initBNPDir=$INIT_STRUCT_DIR/$bnpType
    annealDirName=$SIM_DATA_DIR/$bnpType
    echo "$bnpType Directory:"
    for initBNPPath in $initBNPDir/*lmp; do
        bnpName=$(echo $initBNPPath | awk -F'/' '{print $NF}')
        if [ -d $annealDirName/${bnpName::-4} ]; then echo "  ${bnpName::-4} exists! Skipping..."
        else mkdir $annealDirName/${bnpName::-4}; touch $annealDirName/${bnpName::-4}/$CONFIG_FILE; echo "  ${bnpName::-4} done!"; fi
    done
done
echo -e "Done!\n"

