#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -v NJOBS,NJOB
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a
  
# =============================================================================
#  Self resubmitting PBS bash script:
#
#  * Submits a followon job after executing the current job. Hence it assumes
#    the program will complete within the requested time to allow time for
#    the resubmission
#
#  * Assumes program being run is checkpointing at regular intervals and is
#    able to resume execution from a checkpoint
#
#  * Uses an environment variable (NJOBS) to limit the total number of 
#    resubmissions in the sequence of jobs.
#
#  * Allows the early termination of the sequence of jobs - just create/touch
#    the file STOP_SEQUENCE in the jobs working directory.  This may be done 
#    by the executable program when it has completed the "whole" job or by hand 
#    if there is a problem
#
#  * This script may be renamed anything (<15 characters) but if you use the -N 
#    option to qsub you must edit the qsub line below to give the script name 
#    explicitly
#
#  * To use: 
#         - make appropriate changes to the PBS options above and to the 
#           execution and file manipulation lines belo
#         - submit the job with the appropriate value of NJOBS, eg:
#                    qsub -v NJOBS=5 <scriptname>
#
#  * To kill a job sequence, either touch the file STOP_SEQUENCE or qdel
#    the running job (assuming it isnt resubmitting very fast and you can
#    "catch" one to kill).
#
#  * To test, try  "sleep 30"  as your executable line
#
# ===============================================================================

ECHO=/bin/echo

# These variables are assumed to be set:
#   NJOBS is the total number of jobs in a sequence of jobs (defaults to 1)
#   NJOB is the number of the current job in the sequence (defaults to 1)
  
if [ X$NJOBS == X ]; then export NJOBS=1000; fi
if [ X$NJOB == X ]; then export NJOB=1; fi

# Quick termination of job sequence - look for a specific file 
if [ -f STOP_SEQUENCE ] ; then
    $ECHO  "Terminating sequence at job number $NJOB"
    exit 0
fi

# Pre-job file manipulation goes here ...

# ========================================================================
# .... USER INSERTION OF EXECUTABLE LINE HERE 
cd RCSminLarge/
i=1
for bnp in *xyz; do
    if [[ $i -gt 100 ]]; then break; fi
    zip -6 -u -r ../RCSminLarge.zip $bnp
    mv $bnp ../RCSminLargeDone
    i=$((i+1))
done
cd ..
echo "Done! Check that Ctrl+C wasn't used anytime in between if run in terminal."
# ========================================================================
  
# Check the exit status
errstat=$?
if [ $errstat -ne 0 ]; then
    # A brief nap so PBS kills us in normal termination
    # If execution line above exceeded some limit we want PBS
    # to kill us hard
    sleep 5 
    $ECHO "Job number $NJOB returned an error status $errstat - stopping job sequence."
    exit $errstat
fi

# Check if all files have been zipped; if so, stop self-submission
if [ $i -lt 3 ]; then 
    echo "No folder zipped, exiting."
    exit
fi
# Are we in an incomplete job sequence - more jobs to run ?
if [ $NJOB -lt $NJOBS ]; then
    # Post-job file manipulation (preparing for next job etc) goes here ...

    # Now increment counter and submit the next job
    NJOB=$(($NJOB+1))
    $ECHO "Submitting job number $NJOB in sequence of $NJOBS jobs"
    qsub $PBS_JOBNAME
else
    $ECHO "Finished last job in sequence of $NJOBS jobs"
fi
