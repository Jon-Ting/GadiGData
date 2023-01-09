#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to transfer simulation data from /scratch/ to /g/data/ directory & backup to massdata/
# Author: Jonathan Yik Chang Ting
# Date: 16/1/2021

for bnp in Pt*; do
    if ! grep -q "$bnp" tarredList; then
        echo $bnp
        mdss put $bnp $USER/SimAnneal/CS70+/
        echo $bnp >> tarredList
    fi
done

# Other ways to be explored
# tar -c $bnpType.tar -C $SIM_DATA_DIR $bnpType | pigz -p 48 -1 -c > $bnpType.tar.gz
# netcp -C -l mem=2GB,walltime=10:00:00,other=mdss -z -t tarFiles.tar /g/data/q27/jt5911/tarFiles/ jt5911/
