#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=05:00:00,ncpus=1,mem=8GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

# Goal: Automate the generation of LAMMPS input file for each nanoparticle to be simulated
# Author: Jonathan Yik Chang Ting
# Date: 1/12/2020
# Melting points (K) taken from: {Pt: 2041; Co: 1768; Pd: 1828; Au: 1337}
# size: time/task/mem
#   Stage 0: {20: 1/24/3; 40: 1/24/4; 80: 6/48/6; 150: 20/48/8}
#   Stage 1: {20: 7/48/40; 40: 1/48/60; 80: 6/48/90; 150: 20/48/120}
# To do:
#     - Cite the source for melting points!

STAGE=1
declare -a TYPE_ARR=('L10/' 'L12/' 'CS/' 'RCS/' 'RAL/')
declare -a SIZE_ARR=(20 40 80 150)
declare -a ELEMENT_ARR=('Pt' 'Co' 'Pd' 'Au')
declare -a MELT_TEMP_ARR=(2100 1800 1900 1400)

totalDumps=100  # frame (Stages 0 and 2)
annealDumpRate=10  # K/frame (Stages 1 and 3)
initTemp=300  # K

S0period=500000  # fs
S0therInt=100  # fs
S0dumpInt=$(echo "$S0period/$totalDumps" | bc)  # fs

S1startFrame=$(printf "%08d\n" $S0period)  # fs
heatRate=0.25  # K/ps
S1therInt=500  # fs
S1dumpInt=$(echo "$annealDumpRate/$heatRate*1000" | bc)  # fs

S2period=1000000  # fs
S2therInt=200  # fs
S2dumpInt=$(echo "$S2period/$totalDumps" | bc)  # fs

S3startFrame=$(printf "%08d\n" $S2period)  # fs
coolTemp=300  # K
coolRate=0.2  # K/ps
S3therInt=500  # fs
S3dumpInt=$(echo "$annealDumpRate/$coolRate*1000" | bc)  # fs

SIM_DATA_DIR=/scratch/$PROJECT/$USER
GDATA_DIR=/g/data/$PROJECT/$USER
EAM_DIR=$GDATA_DIR/EAM
TEMPLATE_NAME=$GDATA_DIR/scripts/SimAnneal/annealS$STAGE

echo "Looping through directories:"
echo "-----------------------------------------------"
for ((i=0;i<${#SIZE_ARR[@]};i++)); do
    size=${SIZE_ARR[$i]}
    if [ $size -eq 20 ]; then wallTime=$(printf "%02d\n" 1); numTasks=48
    elif [ $size -eq 40 ]; then wallTime=$(printf "%02d\n" 1); numTasks=48
    elif [ $size -eq 80 ]; then wallTime=$(printf "%02d\n" 6); numTasks=48
    elif [ $size -eq 150 ]; then wallTime=$(printf "%02d\n" 20); numTasks=48
    fi
    maxJobTime=$(echo "$wallTime - 0.2" | bc)  # hr
    timeLimit=$(echo "60*60*$maxJobTime*$numTasks" | bc)  # s
    for ((j=0;j<${#TYPE_ARR[@]};j++)); do
        bnpType=${TYPE_ARR[$j]}
        inpDirName=$GDATA_DIR/InitStruct/BNP/$bnpType
        annealDirName=$SIM_DATA_DIR/SimAnneal/$bnpType
        echo "  $bnpType Directory:"
        for bnpDir in $annealDirName*; do
            
            # Identify the targeted directories
            inpFileName=$(echo $bnpDir | grep -oP "(?<=$bnpType).*")
            if [ $bnpType == 'CS/' ]; then  # CS: Au150THPd40COCS
                if [[ $inpFileName =~ ^[A-Z][a-z]$size[A-Z]{2}[A-Z][a-z][0-9]{2,}[A-Z]{4,}$ ]]; then true; else continue; fi
            elif [ $bnpType == 'L10/' ] || [ $bnpType == 'L12/' ]; then  # L10, L12: CoAu150COL10, CoPd150TOL12
                if [[ $inpFileName =~ ^([A-Z][a-z]){2}$size[A-Z]{2}L1(0|2)$ ]]; then true; else continue; fi
            elif [ $bnpType == 'RAL/' ] || [ $bnpType == 'RCS/' ]; then  # RAL, RCS: CoAu150TO25RAL4, CoPd150TO75RCS6
                if [[ $inpFileName =~ ^([A-Z][a-z]){2}$size[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]$ ]]; then true; else continue; fi
            fi
            echo "    $inpFileName"
            
            # Skip if the input file already exists, otherwise copy template to target directory
            LMP_IN_FILE=${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.in
            # if test -f $LMP_IN_FILE; then echo "      $LMP_IN_FILE exists! Skipping..."; continue; fi
            cp $TEMPLATE_NAME.in ${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.in
            echo "      Scripts copied!"

            # Compute and substitute variables in LAMMPS input file
            elements=($(echo $inpFileName | grep -o "[A-Z][a-z]"))
            element1=${elements[0]}
            element2=${elements[1]}
            potFile=$EAM_DIR/setfl_files/$element1$element2.set
            for ((k=0;k<${#ELEMENT_ARR[@]};k++)); do
                if echo ${elements[@]} | grep -q ${ELEMENT_ARR[$k]}; then heatTemp=${MELT_TEMP_ARR[$k]}; break; else continue; fi
            done
            sed -i "s/{INP_FILE_NAME}/$inpFileName/g" $LMP_IN_FILE
            sed -i "s/{ELEMENT1}/$element1/g" $LMP_IN_FILE
            sed -i "s/{ELEMENT2}/$element2/g" $LMP_IN_FILE
            sed -i "s|{POT_FILE}|$potFile|g" $LMP_IN_FILE
            if [ $STAGE -eq 0 ]; then
                sed -i "s/{TIME_LIMIT}/$timeLimit/g" $LMP_IN_FILE
                sed -i "s|{INP_DIR_NAME}|$inpDirName|g" $LMP_IN_FILE
                sed -i "s/{INIT_TEMP}/$initTemp/g" $LMP_IN_FILE
                sed -i "s/{S0_PERIOD}/$S0period/g" $LMP_IN_FILE
                sed -i "s/{S0_THER_INT}/$S0therInt/g" $LMP_IN_FILE
                sed -i "s/{S0_DUMP_INT}/$S0dumpInt/g" $LMP_IN_FILE
            elif [ $STAGE -eq 1 ]; then
                S1period=$(echo "($heatTemp-300)/$heatRate*1000" | bc)  # fs
                timeLimit=$(echo "$S1period/$S0period*$timeLimit" | bc)  # s
                sed -i "s/{TIME_LIMIT}/$timeLimit/g" $LMP_IN_FILE
                sed -i "s/{S1_START_FRAME}/$S1startFrame/g" $LMP_IN_FILE
                sed -i "s/{INIT_TEMP}/$initTemp/g" $LMP_IN_FILE
                sed -i "s/{HEAT_TEMP}/$heatTemp/g" $LMP_IN_FILE
                sed -i "s/{S1_PERIOD}/$S1period/g" $LMP_IN_FILE
                sed -i "s/{S1_THER_INT}/$S1therInt/g" $LMP_IN_FILE
                sed -i "s/{S1_DUMP_INT}/$S1dumpInt/g" $LMP_IN_FILE
            elif [ $STAGE -eq 2 ]; then
                S1period=$(echo "($heatTemp-300)/$heatRate*1000" | bc)  # fs
                S2startFrame=$(printf "%08d\n" $S1period)  # fs
                timeLimit=$(echo "$S2period/$S0period*$timeLimit" | bc)  # s
                sed -i "s/{TIME_LIMIT}/$timeLimit/g" $LMP_IN_FILE
                sed -i "s/{S2_START_FRAME}/$S2startFrame/g" $LMP_IN_FILE
                sed -i "s/{HEAT_TEMP}/$heatTemp/g" $LMP_IN_FILE
                sed -i "s/{S2_PERIOD}/$S2period/g" $LMP_IN_FILE
                sed -i "s/{S2_THER_INT}/$S2therInt/g" $LMP_IN_FILE 
                sed -i "s/{S2_DUMP_INT}/$S2dumpInt/g" $LMP_IN_FILE
            elif [ $STAGE -eq 3 ]; then
                S3period=$(echo "($heatTemp-300)/$coolRate*1000" | bc)  # fs
                timeLimit=$(echo "$S3period/$S0period*$timeLimit" | bc)  # s
                sed -i "s/{TIME_LIMIT}/$timeLimit/g" $LMP_IN_FILE
                sed -i "s/{S3_START_FRAME}/$S3startFrame/g" $LMP_IN_FILE
                sed -i "s/{HEAT_TEMP}/$heatTemp/g" $LMP_IN_FILE
                sed -i "s/{COOL_TEMP}/$coolTemp/g" $LMP_IN_FILE
                sed -i "s/{S3_PERIOD}/$S3period/g" $LMP_IN_FILE
                sed -i "s/{S3_THER_INT}/$S3therInt/g" $LMP_IN_FILE 
                sed -i "s/{S3_DUMP_INT}/$S3dumpInt/g" $LMP_IN_FILE
            else echo "Variable STAGE: $STAGE unrecognised"; exit 1; fi
            echo "      Variables substituted!"
        done
    done
done
echo -e "Done!\n"

