#!/bin/bash
# Goal: Extract last frame of Stage 2 simulation
# Author: Jonathan Yik Chang Ting
# Date: 15/3/2021

for bnp in L10/AuCo20*; do
    echo $bnp
    cd $bnp/
    tar -xf ${bnp}S2.tar.gz
    cd ${bnp}S2/
    for xyz in *mpiio.xyz; do
        numAtoms=$(head -n1 $xyz)
        numLines=$(echo "$numAtoms+2" | bc)
        tail -n$numLines $xyz > ori$xyz
    done
    cd ..
    rm -rf ${bnp}S2
    cd ..
done
echo "Done!"
