#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=05:00:00,ncpus=1,mem=8GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

# Goal: Automate the generation of configuration setting file for each directory containing nanoparticles to be simulated
# Author: Jonathan Yik Chang Ting
# Date: 31/12/2020

declare -a TYPE_ARR=('L10/' 'L12/' 'CS/' 'RCS/' 'RAL/')
SIM_DATA_DIR=/scratch/$PROJECT/$USER
GDATA_DIR=/g/data/$PROJECT/$USER
CONFIG_FILE=config.yml

echo "Looping through directories:"
echo "-----------------------------------------------"
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    annealDirName=$SIM_DATA_DIR/SimAnneal/$bnpType
    echo "$bnpType Directory:"
    for bnpDir in $annealDirName*/; do
        inpFileName=$(echo $bnpDir | grep -oP "(?<=$bnpType).*")
        if test -f $annealDirName/$inpFileName$CONFIG_FILE; then echo "  $CONFIG_FILE exists in $inpFileName! Skipping..."
        else touch $annealDirName/$inpFileName$CONFIG_FILE; echo "  $inpFileName done!"; fi
    done
done
echo -e "Done!\n"

