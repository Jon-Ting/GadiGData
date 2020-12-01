#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime={{WALLTIME}}:00:00,ncpus={{NUM_TASKS}},mem={{MEM}}GB,jobfs={{JOBFS}}GB,software=lammps
#PBS -l storage=gdata/q27
#PBS -l wd

module load lammps/3Mar2020

# Fixed variables
STAGE=0
SIM_DATA_DIR=/scratch/q27/jt5911
GDATA_DIR=/g/data/q27/jt5911
EAM_DIR=$GDATA_DIR/EAM

# Variables to be substituted
wallTime={{WALLTIME}}  # hr
numTasks={{NUM_TASKS}}
totalDumps={{TOTAL_DUMPS}}
inpDirName=$SIM_DATA_DIR/InitStruct/{{INP_DIR_NAME}}
inpFileName={{INP_FILE_NAME}}
initTemp={{INIT_TEMP}}  # K
S0period={{S0_PERIOD}}  # fs
S0therInt={{S0_THER_INT}}  # fs

# Variables to be computed
elements=($(echo $inpFileName | grep -o "[A-Z][a-z]"))
element1=${elements[0]}
element2=${elements[1]}
potFile=$EAM_DIR/setfl_files/$element1$element2.set
maxJobTime=$(echo "$wallTime - 0.5" | bc)  # hr
timeLimit=$(echo "60*60*$maxJobTime*$numTasks" | bc)  # s
S0dumpInt=$(echo "$S0period/$totalDumps" | bc)  # fs

mpirun -np $numTasks lmp_openmpi -sf opt \
                                 -in ${inpFileName}S$STAGE.in \
                                 -var inpDirName $inpDirName \
                                 -var inpFileName $inpFileName \
                                 -var element1 $element1 \
                                 -var element2 $element2 \
                                 -var potFile $potFile \
                                 -var timeLimit $timeLimit \
                                 -var initTemp $initTemp \
                                 -var S0period $S0period \
                                 -var S0therInt $S0therInt \
                                 -var S0dumpInt $S0dumpInt \
                                 >> ${inpFileName}S$STAGE.log
mkdir ${inpFileName}S$STAGE
mv ${inpFileName}S$STAGE.*.lmp ${inpFileName}S$STAGE
tar -czvf ${inpFileName}S$STAGE.tar.gz ${inpFileName}S$STAGE
rm -rf ${inpFileName}S$STAGE
