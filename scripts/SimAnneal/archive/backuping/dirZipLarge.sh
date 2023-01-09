#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=01:00:00,ncpus=1,mem=20GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to zip up relevant directories in SimAnneal directory
# Author: Jonathan Yik Chang Ting
# Date: 4/5/2021

declare -a TYPE_ARR=('largeCS')
declare -a ELE1_ARR=('Au' 'Co' 'Pd' 'Pt')
declare -a ELE2_ARR=('Pt' 'Pd' 'Co' 'Au')
declare -a SIZE_ARR=(20 40 80)
declare -a SHAPE_ARR=('CO' 'CU'  'DH' 'IC' 'OT' 'RD' 'TH' 'TO')
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
                    echo $bnpType $ele1Type $ele2Type $size $shape
                    zip -6 -li -r $bnpType.zip $bnpType/${ele1Type}150$shape${ele2Type}$size*CS -u -x *log.lammps
                done
            done
        done
    done
done

declare -a TYPE_ARR=('largeRAL')
declare -a ELE1_ARR=('Au' 'Co' 'Pd' 'Pt')
declare -a ELE2_ARR=('Pt' 'Pd' 'Co' 'Au')
declare -a RATIO_ARR=(25 50 75)
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    for ((j=0;j<${#ELE1_ARR[@]};j++)); do
        ele1Type=${ELE1_ARR[$j]}
        for ((k=0;k<${#ELE2_ARR[@]};k++)); do
            ele2Type=${ELE2_ARR[$k]}
            if [ "$ele1Type" == "$ele2Type" ]; then continue; fi
            for ((l=0;l<${#RATIO_ARR[@]};l++)); do
                ratio=${RATIO_ARR[$l]}
                echo $bnpType $ele1Type $ele2Type $ratio
                zip -6 -li -r $bnpType.zip $bnpType/${ele1Type}${ele2Type}150*$ratio* -u -x *log.lammps
            done
        done
    done
done

