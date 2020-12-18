# Goal: Submit jobs efficiently to NCI Gadi following modification of job scripts
# Author: Jonathan Yik Chang Ting
# Date: 15/12/2020

STAGE=0
maxQueueNum=100
numSeqJob=1000
numInQueue=$(qselect -u $USER | wc -l)
numToSub=$(echo "$maxJob-$numInQueue") | bc
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
declare -a SIZE_ARR=(20 40 80 150)
declare -a WALLTIME_ARR=('00:45:00' '01:00:00' '06:00:00' '20:00:00')
declare -a MEM_ARR=(3 4 6 8)

cd $SIM_DATA_DIR
for ((i=0;i<${#SIZE_ARR[@]};i++)); do
    size=${SIZE_ARR[$i]}
    numJobLeft=$(wc -l $SIM_DATA_DIR/jobList$size | awk '{print $1}')
    if [ $numJobLeft -eq 0 ]; then continue; fi
    cp runAnnealTemplate.sh runAnneal$size.sh
    sed -i 's/{WALLTIME}/${WALLTIME_ARR[$x]}/g' runAnneal$size.sh
    sed -i 's/{MEM}/${MEM_ARR[$x]}/g' runAnneal$size.sh
    sed -i 's/{SIZE}/${SIZE_ARR[$x]}/g' runAnneal$size.sh
    for j in {1..$numJobLeft}; do
        qsub -v NJOBS=$numSeqJob $SCRIPT_DIR/runAnneal$size.sh
    done
done

