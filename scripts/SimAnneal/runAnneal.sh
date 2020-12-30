#!/bin/bash
#PBS -P q27
#PBS -q express
#PBS -l ncpus=192,walltime=04:29:41,mem=8GB
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
ECHO=/bin/echo
if [ X$NJOBS == X ]; then $ECHO "NJOBS (total number of jobs in sequence) is not set - defaulting to 100"; export NJOBS=100; fi
if [ X$NJOB == X ]; then $ECHO "NJOB (current job number in sequence) is not set - defaulting to 1"; export NJOB=1; fi
if [ X$jobName == X ]; then $ECHO "jobName is not set"; exit 1; fi
trap ctrl_c SIGINT
function ctrl_c() { echo -e "\nExiting"; rm -f $EXAM_LOCK $RUN_LOCK; ls; exit 1; }

# Program execution
module load lammps/3Mar2020
RUN_LOCK=run.lock
dirName=$(echo ${jobName%/*})
if test -f $dirName/$RUN_LOCK; then echo "$dirName busy!"; exit 1
elif test -f $jobName.log; then
    if [[ $(tail $jobName.log) =~ "DONE!" ]]; then echo "$jobName simulated!"; exit 1; fi
fi
$ECHO "Running $jobName"; cd $dirName; touch $RUN_LOCK
mpirun -np 192 lmp_openmpi -sf opt -in $jobName.in > $jobName.log

# Post execution
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
errstat=$?
if [ $errstat -ne 0 ]; then sleep 5; $ECHO "Job \#$NJOB gave error status $errstat - Stopping sequence"; rm -f $RUN_LOCK; ls; exit $errstat
elif [ ! -f $jobName.mpiio.rst ]; then $ECHO "No checkpoint file $jobName.mpiio.rst"; rm -f $RUN_LOCK; ls; exit 1
elif [[ ! $(tail $jobName.log) =~ "DONE!" ]]; then $ECHO "String DONE not in $jobName.log!"; rm -f $RUN_LOCK; ls; exit 1; fi
$ECHO -e "\nSimulation done, compressing .lmp files..."
mkdir $jobName; mv $jobName.*.lmp $jobName; tar -czvf $jobName.tar.gz $jobName; rm -rf $jobName $RUN_LOCK; ls; cd $SIM_DATA_DIR
if [ -f STOP_SEQUENCE ]; then $ECHO  "Terminating sequence at job number $NJOB"; exit 0
elif [ $NJOB -ge $NJOBS ]; then $ECHO -e "\nFinished last job in sequence of $NJOBS jobs"; exit 0; fi

# Setup for next job
i=0
maxIter=100
EXAM_LOCK=examine.lock
JOB_LIST=jobList
numJobLeft=$(wc -l $SIM_DATA_DIR/$JOB_LIST | awk '{print $1}')
if [ $numJobLeft -eq 0 ]; then echo "No more job in $JOB_LIST"; exit 0; else echo "$numJobLeft job(s) in $JOB_LIST"; fi
while [ $i -lt $maxIter ]; do 
    if test -f $EXAM_LOCK; then sleep 1; i=$[$i+1]
    else touch $EXAM_LOCK; jobName=$(head -n1 $JOB_LIST); tail -n+2 $JOB_LIST > $JOB_LIST.2; mv $JOB_LIST.2 $JOB_LIST || continue; break; fi
done
if [ $i -gt $maxIter ]; then echo "Waited for too long! Check $EXAM_LOCK"; exit 1; fi
NJOB=$(($NJOB+1))
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
initStruct=$(grep read_data $jobName.in | awk '{print $2}')
numAtoms=$(grep atoms $initStruct | awk '{print $1}')
ncpus=$($ECHO "scale=0; ($numAtoms/32000+1) * 24" | bc)
mem=$($ECHO "scale=0; -($numAtoms-320000)*($numAtoms-320000)/40000000000 + 8" | bc)  # GB
wallTime=$($ECHO "(12*$numAtoms+360000) / $ncpus" | bc)  # s
hr=$(printf "%02d\n" $($ECHO "scale=0; $wallTime / 60 / 60" | bc))  # hr
min=$(printf "%02d\n" $($ECHO "scale=0; ($wallTime-$hr*60*60) / 60" | bc))  # min
sec=$(printf "%02d\n" $($ECHO "scale=0; $wallTime - $hr*60*60 - $min*60" | bc))  # s
sed -i "0,/^.*-l ncpus=.*$/s//#PBS -l ncpus=$ncpus,walltime=$hr:$min:$sec,mem=${mem}GB/" $SCRIPT_DIR/$PBS_JOBNAME
sed -i "0,/^.*mpirun.*$/s//mpirun -np $ncpus lmp_openmpi -sf opt -in \$jobName.in > \$jobName.log/" $SCRIPT_DIR/$PBS_JOBNAME
$ECHO -e "\nSubmitting job number $NJOB in sequence of $NJOBS jobs\n$jobName\nnumAtoms,ncpus,walltime,mem = $numAtoms,$ncpus,$hr:$min:$sec,$mem"
qsub $SCRIPT_DIR/$PBS_JOBNAME
rm -f $EXAM_LOCK
