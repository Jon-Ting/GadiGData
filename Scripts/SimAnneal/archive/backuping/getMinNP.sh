#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Extract final and minimised structure of frames captured during Stage 2 simulation into specific directories
# Author: Jonathan Yik Chang Ting
# Date: 22/1/2022

bnpType=RAL
cd $bnpType/
for bnp in Pd*; do
    echo $bnp
    if grep -q $bnp ../checkHere; then continue; fi
    cd $bnp/; rm -rf ${bnp}S2/
    #tar -xf ${bnp}S2.tar.gz
    unzip ${bnp}S2.zip
    if [[ ! -d "${bnp}S2" ]]; then 
        rm -f zi*
        echo $bnp >> ../../toRerunRAL.txt
        cd ..; continue
    fi
    cp -r ${bnp}S2/*min*xyz ../../${bnpType}min/
    cd ${bnp}S2/
    for xyz in *mpiio.xyz; do
        numAtoms=$(head -n1 $xyz)
        numLines=$(echo "$numAtoms+2" | bc)
        tail -n$numLines $xyz > ../../../${bnpType}ori/$xyz
    done
    cd ..; rm -rf ${bnp}S2/; cd ..
    echo $bnp >> ../checkHere
done
echo "Done!"
