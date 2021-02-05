#!/bin/bash
# Goal: Script to generate restart files using MPIIO package from usual restart files
# Author: Jonathan Yik Chang Ting
# Date: 1/1/2021

EAM_DIR=/g/data/$PROJECT/$USER/EAM
TEMPLATE_FILE=/g/data/$PROJECT/$USER/scripts/SimAnneal/genMPIrst.in
for dirName in */; do
    if [ ! -f $dirName/*S0.rst ]; then continue; fi
    cd $dirName; cp $TEMPLATE_FILE .
    inpFileName=${dirName::-1}
    elements=($(echo $inpFileName | grep -o "[A-Z][a-z]")); element1=${elements[0]}; element2=${elements[1]}
    potFile=$EAM_DIR/setfl_files/$element1$element2.set
    sed -i "s/\${INP_FILE_NAME}/$inpFileName/" genMPIrst.in
    sed -i "s|\${POT_FILE}|$potFile|" genMPIrst.in
    sed -i "s/\${ELEMENT1}/$element1/" genMPIrst.in
    sed -i "s/\${ELEMENT2}/$element2/" genMPIrst.in
    mv log.lammps log.lammps1; lmp_openmpi -in genMPIrst.in; mv log.lammps1 log.lammps
    rm -rf *S0.rst genMPIrst.in; cd ..
done
