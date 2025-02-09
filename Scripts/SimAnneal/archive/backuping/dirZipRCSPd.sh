#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to zip up relevant directories in SimAnneal directory
# Author: Jonathan Yik Chang Ting
# Date: 4/5/2021

declare -a TYPE_ARR=('RCS')
declare -a ELE1_ARR=('Au' 'Co' 'Pd' 'Pt')
declare -a ELE1_ARR=('Pd')
declare -a SIZE1_ARR=(20 30 40 50)
declare -a SHAPE1_ARR=('CO' 'CU' 'DH' 'IC' 'OT' 'RD' 'TH' 'TO')
declare -a ELE2_ARR=('Pt' 'Pd' 'Co' 'Au')
#declare -a SIZE2_ARR=(20 30 40 50 60 70)
#declare -a SHAPE2_ARR=('CO' 'CU' 'DH' 'IC' 'OT' 'RD' 'TH' 'TO')
declare -a RATIO_ARR=(25 50 75)
declare -a REP_ARR=(0 1 2 3 4 5 6 7 8 9)
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    for ((j=0;j<${#ELE1_ARR[@]};j++)); do
        ele1Type=${ELE1_ARR[$j]}
        for ((k=0;k<${#SIZE1_ARR[@]};k++)); do
            size1=${SIZE1_ARR[$k]}
            for ((l=0;l<${#SHAPE1_ARR[@]};l++)); do
                shape1=${SHAPE1_ARR[$l]}
                for ((m=0;m<${#ELE2_ARR[@]};m++)); do
                    ele2Type=${ELE2_ARR[$m]}
                    if [ "$ele1Type" == "$ele2Type" ]; then continue; fi
                    for ((n=0;n<${#RATIO_ARR[@]};n++)); do
                        ratio=${RATIO_ARR[$n]}
                        #if [ $size2 -ge $size1 ]; then continue; fi
                        for ((o=0;o<${#REP_ARR[@]};o++)); do
                            rep=${REP_ARR[$o]}
                            echo $ele1Type$ele2Type$size1$shape1$ratio$bnpType$rep
                            #if [[ -f "$ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}.tar.gz" ]]; then
                            #    echo "Tarball exists"
                            #else
                            #    tar -czf $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}.tar.gz $ele1Type$size1$shape1$ele2Type$size2$shape2${bnpType}/
                            #fi
                            if grep -q $ele1Type$ele2Type$size1$shape1$ratio$bnpType$rep zipped$bnpType$ele1Type.txt; then continue; fi
                            zip -6 -li -r $bnpType$ele1Type.zip $bnpType/$ele1Type$ele2Type$size1$shape1$ratio$bnpType$rep -u -x *log.lammps
                            echo $ele1Type$ele2Type$size1$shape1$ratio$bnpType$rep >> zipped$bnpType$ele1Type.txt
                        done
                    done
                done
            done
        done
    done
done
