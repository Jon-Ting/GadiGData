#!/bin/bash
# Goal: Submit jobs efficiently to NCI Gadi following modification of job scripts
# Author: Jonathan Yik Chang Ting
# Date: 15/12/2020

SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
EXAM_LOCK=examine.lock
JOB_LIST=jobList
SCRIPT=runAnneal.sh

maxQueueNum=30
numInQueue=$(qselect -u $USER | wc -l)
numToSub=$(echo "$maxQueueNum-$numInQueue" | bc)
echo -e "maxQueueNum: $maxQueueNum\nnumInQueue: $numInQueue\nnumToSub: $numToSub"
cd $SIM_DATA_DIR
for (( a=0; $a<$numToSub; a++ )); do
    numJobLeft=$(wc -l $SIM_DATA_DIR/$JOB_LIST | awk '{print $1}')
    if [ $numJobLeft -eq 0 ]; then echo -e "\nNo more job in $JOB_LIST"; exit 0; else echo -e "\n$numJobLeft job(s) in $JOB_LIST"; fi
    i=0
    maxIter=100
    while [ $i -lt $maxIter ]; do 
        if test -f $EXAM_LOCK; then sleep 1; i=$[$i+1]
        else
            touch $EXAM_LOCK; jobName=$(head -n1 $JOB_LIST); tail -n+2 $JOB_LIST > $JOB_LIST.2; mv $JOB_LIST.2 $JOB_LIST || continue
            break; fi
    done
    if [ $i -gt $maxIter ]; then echo "Waited for too long! Check $EXAM_LOCK"; exit 1; fi
    initStruct=$(grep read_data $jobName.in | awk '{print $2}')
    numAtoms=$(grep atoms $initStruct | awk '{print $1}')
    ncpus=$(echo "scale=0; ($numAtoms/32000+1) * 24" | bc)
    mem=$(echo "scale=0; -($numAtoms-320000)*($numAtoms-320000)/40000000000 + 8" | bc)  # GB
    wallTime=$(echo "(12*$numAtoms+360000) / $ncpus" | bc)  # s
    hr=$(printf "%02d\n" $(echo "scale=0; $wallTime / 60 / 60" | bc))  # hr
    min=$(printf "%02d\n" $(echo "scale=0; ($wallTime-$hr*60*60) / 60" | bc))  # min
    sec=$(printf "%02d\n" $(echo "scale=0; $wallTime - $hr*60*60 - $min*60" | bc))  # s
    sed -i "0,/^.*-l ncpus=.*$/s//#PBS -l ncpus=$ncpus,walltime=$hr:$min:$sec,mem=${mem}GB/" $SCRIPT_DIR/$SCRIPT
    sed -i "0,/^.*mpirun.*$/s//mpirun -np $ncpus lmp_openmpi -sf opt -in \$jobName.in > \$jobName.log/" $SCRIPT_DIR/$SCRIPT
    qsub -v jobName=$jobName $SCRIPT_DIR/$SCRIPT
    rm -f $EXAM_LOCK
    echo -e "Submitted $jobName\nnumAtoms,ncpus,walltime,mem = $numAtoms,$ncpus,$hr:$min:$sec,$mem"
done
