#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=150GB,jobfs=10GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to zip up relevant directories in SimAnneal directory
# Author: Jonathan Yik Chang Ting
# Date: 4/5/2021

declare -a TYPE_ARR=('L12')
declare -a ELE1_ARR=('Au' 'Co' 'Pd' 'Pt')
declare -a ELE2_ARR=('Pt' 'Pd' 'Co' 'Au')
declare -a SIZE_ARR=(150)
declare -a SHAPE_ARR=('CO' 'CU' 'OT' 'RD' 'TH' 'TO')
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    for ((j=0;j<${#ELE1_ARR[@]};j++)); do
        ele1Type=${ELE1_ARR[$j]}
        for ((k=0;k<${#ELE2_ARR[@]};k++)); do
            ele2Type=${ELE2_ARR[$k]}
            if [ "$ele1Type" == "$ele2Type" ]; then continue; fi
            for ((l=0;l<${#SIZE_ARR[@]};l++)); do
                size=${SIZE_ARR[$l]}
                for ((m=0;m<${#SHAPE_ARR[@]};m++)); do
                    shape=${SHAPE_ARR[$m]}
                    if [[ -f "${bnpType}large/$ele1Type$ele2Type$size$shape${bnpType}.tar.gz" ]]; then
                        echo "File exists"
                    else
                        echo $bnpType $ele1Type $ele2Type $size $shape
                        tar -czf ${bnpType}large/$ele1Type$ele2Type$size$shape${bnpType}.tar.gz ${bnpType}large/$ele1Type$ele2Type$size$shape*
                    fi
                    #zip -6 -li -r $bnpType.zip $bnpType/$ele1Type$ele2Type$size$shape* -u -x *log.lammps
                done
            done
        done
    done
done
