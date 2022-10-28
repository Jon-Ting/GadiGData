SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal; JOB_LIST=jobList; QUEUE_LIST=queueList; RUN_LIST=runList; RUN_LOCK=run.lock
for jobName in $(cat $JOB_LIST); do
    dirName=$(echo ${jobName%/*}); unqName=$(echo $jobName | awk -F'/' '{print $NF}')
    sed -i "/$unqName/d" $SIM_DATA_DIR/$JOB_LIST
    if test -f $dirName/$RUN_LOCK; then
        echo "$dirName already running!"; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST; exit 1
    elif test -f $jobName.log; then
        if [[ $(tail $jobName.log) =~ "DONE!" ]]; then echo "$jobName simulated!"; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST; exit 1
        else echo "Previous $jobName job unfinished, removing log file"; rm -f $jobName.log; fi; fi
    echo "Running $jobName"; cd $dirName; touch $RUN_LOCK
    echo "$jobName ${PBS_JOBNAME::-2}${PBS_JOBID::-9}" >> $SIM_DATA_DIR/$RUN_LIST; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST 
    if grep -q "re" <<< "${jobName: -2}"; then
        initName=${jobName::-2}; nLog=$(ls $initName*.log | wc -l); mv $initName.log ${initName}r$nLog.log
        if grep -q "S1" <<< "${initName: -2}"; then mv $initName.rdf ${initName}r$nLog.rdf; fi
    else initName=$jobName; fi
    #mpirun -np 48 
    lmp_openmpi -sf opt -in $jobName.in > $initName.log
    
    # Post execution
    errstat=$?; tarName=$(echo $initName | awk -F'/' '{print $NF}') 
    if [ $errstat -ne 0 ]; then
        sleep 5; echo "Job \#$NJOB gave error status $errstat - Stopping sequence"
        rm -f $RUN_LOCK; sed -i "/$unqName/d" $SIM_DATA_DIR/$RUN_LIST; ls; exit $errstat
    elif [[ ! $(tail $initName.log) =~ "DONE!" ]]; then
        sleep 5; echo "String DONE not in $initName.log!"
        rm -f $RUN_LOCK; sed -i "/$unqName/d" $SIM_DATA_DIR/$RUN_LIST; ls; exit 1
    fi
    echo -e "\nSimulation done, compressing .lmp files..."
    if grep -q "re" <<< "${jobName: -2}"; then tar -xf $tarName.tar.gz; else mkdir $tarName; fi
    mv $tarName.*.lmp $tarName; tar -czvf $tarName.tar.gz $tarName; rm -rf $tarName $RUN_LOCK; ls; cd $SIM_DATA_DIR
    sed -i "/$unqName/d" $SIM_DATA_DIR/$RUN_LIST 
    sleep 30
done
