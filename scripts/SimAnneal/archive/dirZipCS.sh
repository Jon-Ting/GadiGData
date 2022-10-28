#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to zip up relevant directories in SimAnneal directory
# Author: Jonathan Yik Chang Ting
# Date: 4/5/2021

declare -a TYPE_ARR=('CS')
declare -a ELE1_ARR=('Au' 'Co' 'Pd' 'Pt')
declare -a SIZE1_ARR=(30 40 50 60 70 80)
declare -a SHAPE1_ARR=('CO' 'CU' 'DH' 'IC' 'OT' 'RD' 'TH' 'TO')
declare -a ELE2_ARR=('Pt' 'Pd' 'Co' 'Au')
declare -a SIZE2_ARR=(20 30 40 50 60 70)
declare -a SHAPE2_ARR=('CO' 'CU' 'DH' 'IC' 'OT' 'RD' 'TH' 'TO')
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    #cd $bnpType
    for ((j=0;j<${#ELE1_ARR[@]};j++)); do
        ele1Type=${ELE1_ARR[$j]}
        for ((k=0;k<${#SIZE1_ARR[@]};k++)); do
            size1=${SIZE1_ARR[$k]}
            for ((l=0;l<${#SHAPE1_ARR[@]};l++)); do
                shape1=${SHAPE1_ARR[$l]}
                for ((m=0;m<${#ELE2_ARR[@]};m++)); do
                    ele2Type=${ELE2_ARR[$m]}
                    if [ "$ele1Type" == "$ele2Type" ]; then continue; fi
                    for ((n=0;n<${#SIZE2_ARR[@]};n++)); do
                        size2=${SIZE2_ARR[$n]}
                        if [ $size2 -ge $size1 ]; then continue; fi
                        for ((o=0;o<${#SHAPE2_ARR[@]};o++)); do
                            shape2=${SHAPE2_ARR[$o]}
                            echo $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}
                            #if [[ -f "$ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}.tar.gz" ]]; then
                            #    echo "Tarball exists"
                            #else
                            #    tar -czf $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}.tar.gz $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}/
                            #fi
                            if grep -q $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType} zipped.txt; then continue; fi
                            zip -6 -li -r $bnpType.zip $bnpType/$ele1Type$size1$shape1$ele2Type$size2$shape2* -u -x *log.lammps
                            echo $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType} >> zipped.txt
                        done
                    done
                done
            done
        done
    done
done
