#!/bin/bash
# Script to generate core-shell bimetallic nanoparticles (CS BNPs).
# Feed different parameters into the LAMMPS input template file genBNPCS.in
# Author: Jonathan Yik Chang Ting
# Date: 29/10/2020
# Atomic mass taken from 
# Metallic radii taken from Greenwood, Norman N.; Earnshaw, Alan (1997). Chemistry of the Elements (2nd ed.). Butterworth-Heinemann. ISBN 978-0-08-037941-8

# To do:
# - Add atomic mass reference
# - Validate potential files

# Define variables
EAM_DIR=/g/data/q27/jt5911/EAM/setfl_files
LMP_DATA_DIR=/g/data/q27/jt5911/InitStruct
MNP_DIR=MNP
BNP_DIR=BNP
CS_DIR=CS
IN_TEMPLATE=genBNPCS.in
logFile=lmpBNPCS.log
latType=fcc
declare -a ELEMENT_ARR=('Co' 'Pd' 'Pt' 'Au')
# declare -a ELEMENT_ARR=('Au' 'Pt' 'Pd' 'Co')
# declare -a MASS_ARR=(58.933 106.42 195.08 196.97)
declare -a MASS_ARR=(196.97 195.08 106.42 58.933)
# declare -a RADIUS_ARR=(1.25 1.37 1.39 1.44)
declare -a RADIUS_ARR=(1.44 1.39 1.37 1.25)
# declare -a FCC_LC_ARR=(3.537 3.89 3.92 4.09)
declare -a FCC_LC_ARR=(4.09 3.92 3.89 3.537)
declare -a SIZE_ARR=(20 30 40 50 60 70 80)
declare -a SHAPE_ARR=('CU' 'TH' 'RD' 'OT' 'TO' 'CO' 'DH' 'IC')

echo "Generating core-shell bimetallic nanoparticles:"
echo "-----------------------------------------------"
for ((i=0;i<${#ELEMENT_ARR[@]};i++)); do
    element1=${ELEMENT_ARR[$i]}
    mass1=${MASS_ARR[$i]}
    radius1=${RADIUS_ARR[$i]}
    for ((j=$i+1;j<${#ELEMENT_ARR[@]};j++)); do
        element2=${ELEMENT_ARR[$j]}
        mass2=${MASS_ARR[$j]}
        radius2=${RADIUS_ARR[$j]}
        delCutoff=$(echo "scale=3;$radius1+$radius2" | bc)
        latConst=${FCC_LC_ARR[$j]}
        potFile="$EAM_DIR/$element1$element2.set"
        echo "$element2@$element1 $delCutoff"
        # Check if EAM potential file exists
        if test -f $potFile; then echo "  Using $potFile"; else echo "  $potFile not found!"; fi
        for ((k=1;k<${#SIZE_ARR[@]};k++)); do
            size1=${SIZE_ARR[$k]}
            for ((l=0;l<$k;l++)); do
                size2=${SIZE_ARR[$l]}
                for ((m=0;m<${#SHAPE_ARR[@]};m++)); do
                    shape1=${SHAPE_ARR[$m]}
                    fileName1="$LMP_DATA_DIR/$MNP_DIR/$element1$size1$shape1.lmp"
                    xBoxSize1=$(grep 'xlo xhi' $fileName1 | awk '{print $2}')
                    yBoxSize1=$(grep 'ylo yhi' $fileName1 | awk '{print $2}')
                    zBoxSize1=$(grep 'zlo zhi' $fileName1 | awk '{print $2}')
                    for ((n=0;n<${#SHAPE_ARR[@]};n++)); do
                        shape2=${SHAPE_ARR[$n]}
                        fileName2="$LMP_DATA_DIR/$MNP_DIR/$element2$size2$shape2.lmp"
                        outFile="$LMP_DATA_DIR/$BNP_DIR/$CS_DIR/$element1$size1$shape1$element2$size2${shape2}CS.lmp"
                        # Check if the BNP has been generated
                        if test -f $outFile; then
                            echo "  $element1$size1$shape1$element2$size2${shape2}CS.lmp already generated, skipping...";
                            continue
                        else
                            echo "  Generating $element1$size1$shape1$element2$size2${shape2}CS.lmp..."; fi
                        # Compute shifts in atoms when reading the core MNP
                        xBoxSize2=$(grep 'xlo xhi' $fileName2 | awk '{print $2}')
                        yBoxSize2=$(grep 'ylo yhi' $fileName2 | awk '{print $2}')
                        zBoxSize2=$(grep 'zlo zhi' $fileName2 | awk '{print $2}')
                        xShift=$(echo "scale=3;($xBoxSize1-$xBoxSize2)/2" | bc)
                        yShift=$(echo "scale=3;($yBoxSize1-$yBoxSize2)/2" | bc)
                        zShift=$(echo "scale=3;($zBoxSize1-$zBoxSize2)/2" | bc)
                        lmp_openmpi -in $IN_TEMPLATE \
                                    -var element1 $element1 \
                                    -var element2 $element2 \
                                    -var mass1 $mass1 \
                                    -var mass2 $mass2 \
                                    -var fileName1 $fileName1 \
                                    -var fileName2 $fileName2 \
                                    -var xShift $xShift \
                                    -var yShift $yShift \
                                    -var zShift $zShift \
                                    -var latType $latType \
                                    -var latConst $latConst \
                                    -var delCutoff $delCutoff \
                                    -var potFile $potFile \
                                    -var outFile $outFile \
                                    >> $logFile
                        if [ $(grep -c 'DONE!' log.lammps) -eq 0 ]; then echo "  Error!"; else echo "  Done!"; fi
                    done
                done
            done
        done
    done
done
echo -e "Done!\n"

