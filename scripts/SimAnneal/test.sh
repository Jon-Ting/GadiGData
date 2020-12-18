#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=24,walltime={WALLTIME},mem={MEM}GB,storage=scratch/q27+gdata/q27
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
if [ X$NJOBS == X ]; then $ECHO "NJOBS (total number of jobs in sequence) is not set - defaulting to 1"; export NJOBS=1; fi
if [ X$NJOB == X ]; then $ECHO "NJOB (current job number in sequence) is not set - defaulting to 1"; export NJOB=1; fi
trap ctrl_c SIGINT
function ctrl_c() { echo -e "\nExiting"; rm -f $EXAM_LOCK $RUN_LOCK; ls; exit 1; }

# Program execution
module load lammps/3Mar2020
numTasks=24
RUN_LOCK=run.lock
dirName=$(echo ${jobName%/*})
if test -f $dirName/$RUN_LOCK; then echo "$dirName busy!"; exit 1
elif test -f $jobName.log; then if [[ $(tail $jobName.log) =~ "DONE!" ]]; then echo "$jobName simulated!"; exit 1; fi
$ECHO "Running $jobName"; cd $dirName; touch $RUN_LOCK
mpirun -np $numTasks lmp_openmpi -sf opt -in $jobName.in > $jobName.log

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
while [ $i -lt $maxIter ]; do 
    if test -f $EXAM_LOCK; then sleep 1; i=$[$i+1]
    else touch $EXAM_LOCK; jobName=$(head -n1 $JOB_LIST); tail -n+2 $JOB_LIST > $JOB_LIST.tmp; mv $JOB_LIST.tmp $JOB_LIST || continue; break; fi
done
if [ $i -gt $maxIter ]; then echo "Waited for too long! Exiting"; exit 1; fi
NJOB=$(($NJOB+1))
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
initStruct=$(grep read_data *S0.in | awk '{print $2}')
numAtoms=$(grep atoms $initStruct | awk '{print $1}')
wallTime=$($ECHO "$numAtoms" | bc)  # Find the fitting function!
ncpus=($ECHO "$wallTime" | bc)  # Find the function
mem=($ECHO "$numAtoms" | bc)  # Find the fitting function!
cp $SCRIPT_DIR/runAnnealTemplate.sh $SCRIPT_DIR/runAnneal.sh
sed -i 's/{NCPUS}/$ncpus/g' runAnneal.sh
sed -i 's/{WALLTIME}/$wallTime/g' runAnneal.sh
sed -i 's/{MEM}/$mem/g' runAnneal.sh
$ECHO -e "\nSubmitting job number $NJOB in sequence of $NJOBS jobs"
qsub $SCRIPT_DIR/runAnneal.sh -v jobName=$jobName
rm -f $EXAM_LOCK
