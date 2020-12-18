#!/bin/bash
# Goal: Script to generate and update the list of LAMMPS simulations to be run
# Author: Jonathan Yik Chang Ting
# Date: 14/12/2020
# Note: Only applicable to S0 for now

JOB_LIST_FILE=jobList
RUN_LOCK=run.lock
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
for inFile in $SIM_DATA_DIR/*/*/*S0.in; do
    jobPath=${inFile::-3}
    dirName=$(echo ${jobPath%/*})
    if grep -Fx $jobPath $SIM_DATA_DIR/$JOB_LIST_FILE; then continue
    elif test -f $dirName/$RUN_LOCK; then echo "$dirName running a job"; continue
    elif test -f $jobPath.log; then
        if [[ $(tail $jobPath.log) =~ "DONE!" ]]; then echo "$jobPath simulated"; echo $jobPath >> $SIM_DATA_DIR/doneList; continue
        else echo "$jobPath unfinished"; fi
    else echo "$jobPath ready"; fi
    echo $jobPath >> $SIM_DATA_DIR/$JOB_LIST_FILE
done
echo "$JOB_LIST_FILE generated!"
