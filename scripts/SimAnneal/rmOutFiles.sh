# Goal: Clean working directory by removing standard error and output files generated from submitted job that completed fine
# Author: Jonathan Yik Chang Ting
# Date: 10/2/2021


SimDirPath=/scratch/q27/jt5911/SimAnneal
RUN_LOCK=run.lock; RUN_LIST=runList
for stdErrFile in $SimDirPath/runAnneal.sh.e*; do
    sed -i "/^.*Loading lammps.*$/d" $stdErrFile
    sed -i "/^.*Loading requirement.*$/d" $stdErrFile
    sed -i "/^.*mv: cannot stat.*$/d" $stdErrFile
    sed -i "/^.*mkdir: cannot create.*$/d" $stdErrFile
    stdOutFile=$SimDirPath/runAnneal.sh.o${stdErrFile: -8}
    jobPath=$(head -n1 $stdOutFile | awk -F' ' '{print $NF}')
    jobName=$(echo $jobPath | awk -F'/' '{print $NF}')
    if [[ -s $stdErrFile ]]; then
        echo "Non standard lines detected, check $jobName"
        head -n3 $stdErrFile
        if grep -q "PBS: job killed: walltime" $stdErrFile; then
            if grep -q "re" <<< "${jobName: -2}"; then initName=${jobName::-2}; else initName=$jobName; fi
            if grep -q "S2" <<< "${initName: -2}"; then
                dirName=$(echo ${jobPath%/*}); cd $dirName; rm -f $RUN_LOCK
                tac $initName.log | sed '0,/-----------------------------------------------------------------------------------/{/-----------------------------------------------------------------------------------/!d}' | tac >> ${initName}r1.log
                mv ${initName}r1.log $initName.log
                sed -i "/^.*PBS: job killed: walltime.*$/d" $stdErrFile
                sed -i "/^.*${stdErrFile: -8}.*$/d" $SimDirPath/$RUN_LIST
            fi
        fi
    else
        echo "No problem, removing files for $jobName"
        rm -f $stdErrFile $stdOutFile
    fi
done
echo "Done!"
