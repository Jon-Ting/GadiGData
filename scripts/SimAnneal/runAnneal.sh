#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=144,walltime=48:00:00,mem=72GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd
#PBS -v NJOBS,NJOB,jobName
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

# =============================================================================
#  Self resubmitting PBS bash script:
#  * Submits a followon job after the current job. 
#  * Assumes:
#    - completion within requested time to allow time for resubmission.
#    - program is checkpointing regularly & can resume execution from one.
#  * To use: 
#    - change PBS options, execution and file manipulation lines, then
#    *** qsub -v NJOBS=5 jobSeq.sh ***
#  * Varibles:
#    - NJOBS = total number of jobs in a sequence of jobs (default = 1)
#    - NJOB = number of the current job in the sequence (default = 1)
#  * The script may be renamed to anything <15 characters but if -N is used
#    to qsub, script name must be explicitly given to qsub command
#  * To kill job sequence:
#    - touch STOP_SEQUENCE in working directory (by program/hand), or
#    - qdel the running job. 
#  * To test:
#    - try "sleep 30" as executable line
# ===============================================================================

# Initial setup
EXAM_LOCK=examine.lock; RUN_LOCK=run.lock
if [ X$NJOBS == X ]; then echo "NJOBS (total number of jobs in sequence) is not set - defaulting to 100"; export NJOBS=100; fi
if [ X$NJOB == X ]; then echo "NJOB (current job number in sequence) is not set - defaulting to 1"; export NJOB=1; fi
if [ X$jobName == X ]; then echo "jobName is not set"; exit 1; fi
trap ctrl_c SIGINT
function ctrl_c() { echo -e "\nExiting"; rm -f $EXAM_LOCK $RUN_LOCK; ls; exit 1; }

# Program execution
module load lammps/3Mar2020
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
QUEUE_LIST=queueList; RUN_LIST=runList
dirName=$(echo ${jobName%/*}); unqName=$(echo $jobName | awk -F'/' '{print $NF}')
if test -f $dirName/$RUN_LOCK; then echo "$dirName already running!"; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST; exit 1
elif test -f $jobName.log; then
    if [[ $(tail $jobName.log) =~ "DONE!" ]]; then echo "$jobName simulated!"; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST; exit 1
    else echo "Previous $jobName job unfinished"; rm -f $jobName.log; fi; fi
echo "Running $jobName"; cd $dirName; touch $RUN_LOCK
echo $jobName >> $SIM_DATA_DIR/$RUN_LIST
echo $unqName; sed -i "/$unqName/d" $SIM_DATA_DIR/$QUEUE_LIST 
if grep -q "re" <<< "${jobName: -2}"; then
    initName=${jobName::-2}; nLog=$(ls $initName*.log | wc -l); mv $initName.log ${initName}r$nLog.log
    if grep -q "S1" <<< "${initName: -2}"; then mv $initName.rdf ${initName}r$nLog.rdf; fi
else initName=$jobName; fi
mpirun -np 144 lmp_openmpi -sf opt -in $jobName.in > $initName.log

# Post execution
JOB_LIST=jobList; errstat=$?
if [ $errstat -ne 0 ]; then sleep 5; echo "Job \#$NJOB gave error status $errstat - Stopping sequence"; rm -f $RUN_LOCK; ls; exit $errstat
# elif [ ! -f $initName.mpiio.rst ]; then echo "No checkpoint file $initName.mpiio.rst"; rm -f $RUN_LOCK; ls; exit 1
elif [[ ! $(tail $initName.log) =~ "DONE!" ]]; then echo "String DONE not in $initName.log!"; rm -f $RUN_LOCK; ls; exit 1; fi
echo -e "\nSimulation done, compressing .lmp files..."
tarName=$(echo $initName | awk -F'/' '{print $NF}') 
if grep -q "re" <<< "${jobName: -2}"; then tar -xf $tarName.tar.gz; echo "extracted"; ls; else mkdir $tarName; fi
mv $tarName.*.lmp $tarName; tar -czvf $tarName.tar.gz $tarName; rm -rf $tarName $RUN_LOCK; ls; cd $SIM_DATA_DIR
sed -i "/$unqName/d" $SIM_DATA_DIR/$RUN_LIST 
numJobLeft=$(wc -l $SIM_DATA_DIR/$JOB_LIST | awk '{print $1}')
if [ $numJobLeft -eq 0 ]; then echo "No more job in $JOB_LIST"; exit 0; else echo "$numJobLeft job(s) in $JOB_LIST"; fi
if [ -f STOP_SEQUENCE ]; then echo "Terminating sequence at job number $NJOB"; exit 0
elif [ $NJOB -ge $NJOBS ]; then echo -e "\nFinished last job in sequence of $NJOBS jobs"; exit 0; fi

# Setup for next job
i=0; maxIter=100
while [ $i -lt $maxIter ]; do 
    if test -f $EXAM_LOCK; then sleep 1; i=$[$i+1]
    else touch $EXAM_LOCK; jobName=$(head -n1 $JOB_LIST); tail -n+2 $JOB_LIST > $JOB_LIST.2; mv $JOB_LIST.2 $JOB_LIST || continue; break; fi
done
if [ $i -gt $maxIter ]; then echo "Waited for too long! Check $EXAM_LOCK"; exit 1; fi
NJOB=$(($NJOB+1)); SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
if grep -q "re" <<< "${jobName: -2}"; then initName=${jobName::-2}; else initName=$jobName; fi
initStruct=$(grep read_data ${initName::-1}0.in | awk '{print $2}')
numAtoms=$(grep atoms $initStruct | awk '{print $1}')
ncpus=$(echo "scale=0; (($numAtoms-1)/64000+1) * 48" | bc)
numNode=$(echo "scale=0; ($ncpus-1)/48 + 1" | bc)
mem=$(echo "scale=0; (-($numAtoms-320000)*($numAtoms-320000)/40000000000+8) * $numNode" | bc)  # GB (for S0 only at the moment)
wallTime=$(echo "(36*$numAtoms+360000) / $ncpus" | bc)  # s
if grep -q "S1" <<< "${initName: -2}"; then
    mem=$(echo "scale=0; (($mem*0.6)+1) / 1" | bc); wallTime=$(echo "scale=0; ($wallTime*1.5) / 1" | bc); echo "Multiplied mem & wallTime!"
    if [ $wallTime -gt 172800 ]; then wallTime=172800; echo "Limited wallTime!"; fi
fi
hr=$(printf "%02d\n" $(echo "scale=0; $wallTime / 60 / 60" | bc))  # hr
min=$(printf "%02d\n" $(echo "scale=0; ($wallTime-$hr*60*60) / 60" | bc))  # min
sec=$(printf "%02d\n" $(echo "scale=0; $wallTime - $hr*60*60 - $min*60" | bc))  # s
sed -i "0,/^.*-l ncpus=.*$/s//#PBS -l ncpus=$ncpus,walltime=$hr:$min:$sec,mem=${mem}GB/" $SCRIPT_DIR/$PBS_JOBNAME
sed -i "0,/^.*mpirun.*$/s//mpirun -np $ncpus lmp_openmpi -sf opt -in \$jobName.in > \$initName.log/" $SCRIPT_DIR/$PBS_JOBNAME
echo -e "\nSubmitting job number $NJOB in sequence of $NJOBS jobs\n$jobName\nnumAtoms,ncpus,walltime,mem = $numAtoms,$ncpus,$hr:$min:$sec,$mem"
qsub $SCRIPT_DIR/$PBS_JOBNAME; echo $jobName >> $QUEUE_LIST; rm -f $EXAM_LOCK