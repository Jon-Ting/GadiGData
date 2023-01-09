#!/bin/bash

for xyz in *mpiio.xyz; do
    numAtoms=$(head -n1 $xyz)
    numLines=$(echo "$numAtoms+2" | bc)
    tail -n$numLines $xyz > ../../../minStruct/RCSoriLarge/$xyz
    echo $xyz
done
