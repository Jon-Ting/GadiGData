#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus={NCPUS},walltime={WALLTIME},mem={MEM}GB,storage=scratch/q27+gdata/q27
#PBS -l wd
#PBS -v NJOBS,NJOB
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

# NJOBS=5  # DEBUG
ECHO=/bin/echo
if [ X$NJOBS == X ]; then $ECHO "NJOBS (total number of jobs in sequence) is not set - defaulting to 1"; export NJOBS=1; fi
if [ X$NJOB == X ]; then $ECHO "NJOB (current job number in sequence) is not set - defaulting to 1"; export NJOB=1; fi
if [ -f STOP_SEQUENCE ]; then $ECHO  "Terminating sequence at job number $NJOB"; exit 0; fi
trap ctrl_c SIGINT
function ctrl_c() { echo -e "\nExiting"; rm -f $EXAM_LOCK $RUN_LOCK; ls; exit 1; }

# Program execution
module load lammps/3Mar2020

numTasks={NCPUS}
EXAM_LOCK=examine.lock
RUN_LOCK=run.lock
JOB_LIST_FILE=jobList{SIZE}
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
maxIter=100
i=0
while [ $i -lt $maxIter ]; do
    if test -f $EXAM_LOCK; then sleep 1; i=$[$i+1]
    else
        # Extracting path to the job
        touch $EXAM_LOCK
        jobName=$(head -n 1 $JOB_LIST_FILE)
        nextJob=$(tail -n +2  $JOB_LIST_FILE | head -n 1)
        tail -n +3 $JOB_LIST_FILE > $JOB_LIST_FILE.tmp
        mv $JOB_LIST_FILE.tmp $JOB_LIST_FILE || continue
        rm -f $EXAM_LOCK
        # Go there and run simulation 
        dirName=$(echo ${jobName%/*})
        $ECHO "Running $jobName"; cd $dirName; touch $RUN_LOCK
        mpirun -np $numTasks lmp_openmpi -sf opt -in $jobName.in > $jobName.log
        break
    fi
    if [ $i -eq $maxIter ]; then echo "Waited for too long! Exiting"; exit 1; fi
done

errstat=$?
if [ $errstat -ne 0 ]; then sleep 5; $ECHO "Job \#$NJOB gave error status $errstat - Stopping sequence"; rm -f $RUN_LOCK; ls; exit $errstat; fi
if [ $NJOB -lt $NJOBS ]; then
    if [ ! -f $jobName.mpiio.rst ]; then $ECHO "No checkpoint file $jobName.mpiio.rst"; rm -f $RUN_LOCK; ls; exit 1; fi
    if [[ ! $(tail $jobName.log) =~ "DONE!" ]]; then $ECHO "String DONE not in $jobName.log!"; ls; exit 1; fi
    $ECHO -e "\nSimulation done, compressing .lmp files..."
    mkdir $jobName; mv $jobName.*.lmp $jobName; tar -czvf $jobName.tar.gz $jobName; rm -rf $jobName $RUN_LOCK;
    ls; cd $SIM_DATA_DIR

    NJOB=$(($NJOB+1))
    $ECHO -e "\nSubmitting job number $NJOB in sequence of $NJOBS jobs"
    qsub $SCRIPT_DIR/$PBS_JOBNAME
else $ECHO -e "\nFinished last job in sequence of $NJOBS jobs"; fi
