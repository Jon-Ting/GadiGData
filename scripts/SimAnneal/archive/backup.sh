#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=50GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to transfer simulation data from /scratch/ to /g/data/ directory & backup to massdata/
# Author: Jonathan Yik Chang Ting
# Date: 16/1/2021

declare -a TYPE_ARR=('L10' 'L12' 'CS' 'RCS' 'RAL')
declare -a TYPE_ARR=('RAL')
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
GDATA_DIR=/g/data/$PROJECT/$USER

cd $SIM_DATA_DIR
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    # zip -6 -lf $GDATA_DIR/jobLogs/SimAnneal/zip.log -li -r $bnpType.zip $bnpType/ -qTu -x *log.lammps
    mdss put $bnpType.zip $USER/SimAnneal/

    # Other ways to be explored
    # tar -c $bnpType.tar -C $SIM_DATA_DIR $bnpType | pigz -p 48 -1 -c > $bnpType.tar.gz
    # netcp -C -l mem=2GB,walltime=10:00:00,other=mdss -z -t tarFiles.tar /g/data/q27/jt5911/tarFiles/ jt5911/
done
