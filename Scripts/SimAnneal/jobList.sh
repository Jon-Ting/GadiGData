#!/bin/bash
# Goal: Script to generate and update the list of LAMMPS simulations to be run
# Author: Jonathan Yik Chang Ting
# Date: 14/12/2020

STAGE=$1
JOB_LIST_FILE=jobList; QUEUE_LIST_FILE=queueList
CONFIG_FILE=config.yml; RUN_LOCK=run.lock; EXAM_LOCK=examine.lock
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
for inFile in $SIM_DATA_DIR/*/*/*S$STAGE.in; do
    jobPath=${inFile::-3}; dirPath=$(echo ${jobPath%/*}); unqName=$(echo $jobPath | awk -F'/' '{print $NF}')
    if [ $STAGE = 1 ]; then
        eqState=$(grep S0eq: $dirPath/$CONFIG_FILE)
        if ! grep -q "true" <<< "$eqState"; then echo "$jobPath unequilibrated, skipping..."; continue; fi  # Skip if unequilibrated
    elif [ $STAGE = 2 ]; then
        heatState=$(grep S1ok: $dirPath/$CONFIG_FILE)
        if ! grep -q "true" <<< "$heatState"; then echo "$jobPath unmelted, skipping..."; continue; fi  # Skip if unmelted
    fi
    if grep -Fq $jobPath $SIM_DATA_DIR/$JOB_LIST_FILE; then echo "$jobPath on list, skipping..."; continue  # On the to-be-submitted-list
    elif grep -Fq $jobPath $SIM_DATA_DIR/$QUEUE_LIST_FILE; then echo "$jobPath queuing, skipping..."; continue  # Has been submitted
    elif test -f $dirPath/$RUN_LOCK; then echo "$dirPath running a job, skipping..."; continue  # Running
    elif test -f $jobPath.log; then  # Log file exists but not running
        if [[ $(tail -n 350 $jobPath.log) =~ "DONE!" ]]; then
            if [ $STAGE = 2 ]; then
                if [[ ! $(tail -n 5 $jobPath.log) =~ "ALL DONE!" ]]; then
                    doneNum=$(ls $jobPath/*min*xyz | wc -l)
                    echo "$jobPath unfinished, generating job script..."; cp ${jobPath}.in ${jobPath}re.in; jobPath=${jobPath}re
                    sed -i "0,/^.*runNum loop.*$/s//variable        remainNum equal 101-$doneNum\nvariable        loopNum loop \${remainNum}\nvariable        runNum equal \${loopNum}+$doneNum/" $jobPath.in
                    sed -i "0,/next            runNum/s//next            loopNum/" $jobPath.in
                    sed -i "0,/jump.*$/s//jump            SELF/" $jobPath.in
                else
                    runState=$(grep S2ok: $dirPath/$CONFIG_FILE)
                    if ! grep -q "true" <<< "$runState"; then echo "S2ok: true" >> $dirPath/$CONFIG_FILE; fi
                    echo "$jobPath done, skipping..."; continue
                fi
            elif [ $STAGE = 1 ]; then
                if [[ $(tail -n 40 $jobPath.log) =~ "halt timeLimit" ]]; then
                    echo "$jobPath unfinished, generating job script..."; cp ${jobPath}.in ${jobPath}re.in; jobPath=${jobPath}re
                    totSteps=$(echo "$(grep "dumpS1 all" $jobPath.in | awk '{print $5}') * 100" | bc)
                    timeStep=$(echo "$(ls ${jobPath::-2}*.mpiio.rst | wc -l) * 100000" | bc)
                    remSteps=$(echo "$totSteps - $timeStep" | bc)
                    currTemp=$(echo "300 + $timeStep/1000" | bc)
                    sed -i "/^.*read_restart.*$/d" $jobPath.in
                    sed -i "0,/^.*reset.*$/s//read_restart    $unqName.$timeStep.mpiio.rst/" $jobPath.in
                    sed -i "s/nvt temp [0-9]\+/nvt temp $currTemp/" $jobPath.in
                    sed -i "0,/^.*run.*$/s//run             $remSteps/" $jobPath.in
                else
                    heatState=$(grep S1ok: $dirPath/$CONFIG_FILE)
                    if ! grep -q "true" <<< "$heatState"; then echo "S1ok: true" >> $dirPath/$CONFIG_FILE; fi
                    echo "$jobPath done, skipping..."; continue
                fi
            elif [ $STAGE = 0 ]; then
                #echo "$jobPath done (tmp skip)"; continue
                eqState=$(grep S0eq: $dirPath/$CONFIG_FILE)
                if grep -q "true" <<< "$eqState"; then echo "$jobPath equilibrated, skipping..."; continue; fi  # Skip if equilibrated
                bnpType=$(echo $jobPath | awk -F'/' '{print $6}'); dirName=$(echo $jobPath | awk -F'/' '{print $7}')
                python3 vis.py $bnpType $dirName
                eqState=$(grep S0eq: $dirPath/$CONFIG_FILE)
                if grep -q "true" <<< "$eqState"; then echo "$jobPath equilibrated, skipping..."; continue
                elif grep -q "false" <<< "$eqState"; then
                    echo "$jobPath unequilibrated, generating job script..."; cp ${jobPath}.in ${jobPath}re.in; jobPath=${jobPath}re
                    sed -i "/change/d" $jobPath.in; sed -i "/MIN/,+4d" $jobPath.in; sed -i "/reset/,+1d" $jobPath.in
                    sed -i "0,/^.*read_data.*$/s//read_restart    ${dirName}S$STAGE.mpiio.rst/" $jobPath.in
                    sed -i "0,/^.*pe0 equal.*$/s//variable        pe0 equal c_peAll/" $jobPath.in
                    sed -i "0,/^.*SELF break.*$/s//if              \"\${peVar} < 0.5\" then \"jump SELF break\"/" $jobPath.in
                fi 
            fi
        else echo "$jobPath unfinished, submitting..."; fi
    else echo "$jobPath ready, submitting..."; fi
    touch $EXAM_LOCK; echo $jobPath >> $SIM_DATA_DIR/$JOB_LIST_FILE; rm -f $EXAM_LOCK
done
echo "$JOB_LIST_FILE generated!"
