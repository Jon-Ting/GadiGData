#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=48,walltime=03:00:00,mem=40GB,storage=scratch/q27+gdata/q27
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
STAGE=1
TARGET_SIZE=20
declare -a TYPE_ARR=('L10/')

ECHO=/bin/echo
if [ X$NJOBS == X ]; then $ECHO "NJOBS (total number of jobs in sequence) is not set - defaulting to 1"; export NJOBS=1; fi
if [ X$NJOB == X ]; then $ECHO "NJOB (current job number in sequence) is not set - defaulting to 1"; export NJOB=1; fi
if [ -f STOP_SEQUENCE ]; then $ECHO  "Terminating sequence at job number $NJOB"; exit 0; fi
trap ctrl_c SIGINT
function ctrl_c() { echo -e "\nExiting"; rm -f $EXAM_LOCK $RUN_LOCK; ls; exit 1; }

# Program execution
module load lammps/3Mar2020

# if [ $TARGET_SIZE == '20' ] || [ $TARGET_SIZE == '40' ]; then numTasks=24; else numTasks=48; fi
numTasks=48
EXAM_LOCK=EXAMINING
RUN_LOCK=ON_THE_WAY
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
$ECHO "Looping through BNP directories:"
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    annealDirName=$SIM_DATA_DIR/$bnpType
    $ECHO "  $bnpType Directory:"
    for bnpDir in $annealDirName*; do
        inpFileName=$($ECHO $bnpDir | grep -oP "(?<=$bnpType).*")
        if [ $bnpType == 'CS/' ]; then  # CS: Au150THPd40COCS
            if [[ $inpFileName =~ ^[A-Z][a-z]$TARGET_SIZE[A-Z]{2}[A-Z][a-z][0-9]{2,}[A-Z]{4,}$ ]]; then true; else continue; fi
        elif [ $bnpType == 'L10/' ] || [ $bnpType == 'L12/' ]; then  # L10, L12: CoAu150COL10, CoPd150TOL12
            if [[ $inpFileName =~ ^([A-Z][a-z]){2}$TARGET_SIZE[A-Z]{2}L1(0|2)$ ]]; then true; else continue; fi
        elif [ $bnpType == 'RAL/' ] || [ $bnpType == 'RCS/' ]; then  # RAL, RCS: CoAu150TO25RAL4, CoPd150TO75RCS6
            if [[ $inpFileName =~ ^([A-Z][a-z]){2}$TARGET_SIZE[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]$ ]]; then true; else continue; fi
        fi
        if test -f $bnpDir/$EXAM_LOCK; then $ECHO "    $inpFileName examining, skipping..."; continue; else touch $bnpDir/$EXAM_LOCK; cd $bnpDir; fi
        if test -f $RUN_LOCK; then $ECHO "    $inpFileName running, skipping..."; rm -f $EXAM_LOCK; continue
        elif test -f *S$STAGE.log; then
            if [[ $(tail *.log) =~ "DONE!" ]]; then $ECHO "    $inpFileName simulated, skipping..."; rm -f $EXAM_LOCK; continue
            else $ECHO "    $inpFileName incomplete, simulating..."; fi
        else $ECHO "    $inpFileName ready, simulating..."; fi
        touch $RUN_LOCK; rm -f $EXAM_LOCK
        mpirun -np $numTasks lmp_openmpi -sf opt -in $annealDirName/$inpFileName/${inpFileName}S$STAGE.in > ${inpFileName}S$STAGE.log
        break
    done
    if test -f $RUN_LOCK; then break; fi
done

errstat=$?
if [ $errstat -ne 0 ]; then sleep 5; $ECHO "Job \#$NJOB gave error status $errstat - stopping sequence..."; rm -f $RUN_LOCK; ls; exit $errstat; fi
if [ $NJOB -lt $NJOBS ]; then
    if [ ! -f ${inpFileName}S$STAGE.mpiio.rst ]; then $ECHO "No checkpoint file ${inpFileName}S$STAGE.mpiio.rst"; rm -f $RUN_LOCK; ls; exit 1; fi
    if [[ ! $(tail ${inpFileName}S$STAGE.log) =~ "DONE!" ]]; then $ECHO "String DONE not in ${inpFileName}S$STAGE.log!"; ls; exit 1; fi
    $ECHO -e "\nSimulation done, compressing .lmp files..."
    mkdir ${inpFileName}S$STAGE; mv ${inpFileName}S$STAGE.*.lmp ${inpFileName}S$STAGE
    tar -czvf ${inpFileName}S$STAGE.tar.gz ${inpFileName}S$STAGE; rm -rf ${inpFileName}S$STAGE $RUN_LOCK; ls
    cd $SIM_DATA_DIR

    NJOB=$(($NJOB+1))
    $ECHO -e "\nSubmitting job number $NJOB in sequence of $NJOBS jobs"
    qsub $SCRIPT_DIR/$PBS_JOBNAME
else $ECHO -e "\nFinished last job in sequence of $NJOBS jobs"; fi
