#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime={{WALLTIME}}:00:00,ncpus={{NUM_TASKS}},mem={{MEM}}GB,jobfs={{JOBFS}}GB,software=lammps
#PBS -l storage=gdata/q27
#PBS -l wd

module load lammps/3Mar2020

# Fixed variables
STAGE=1
DATA_DIR=/g/data/q27/jt5911
# DATA_DIR=/mnt/c/Users/User/Documents/PhD/Workstation/Data  # DEBUG
EAM_DIR=$DATA_DIR/EAM
IN_TEMPLATE=$DATA_DIR/scripts/SimAnneal/annealS$STAGE.in

# Variables to be substituted
wallTime={{WALLTIME}}  # hr
numTasks={{NUM_TASKS}}
totalDumps={{TOTAL_DUMPS}}
inpFileName={{INP_FILE_NAME}}
initTemp={{INIT_TEMP}}  # K
heatTemp={{HEAT_TEMP}} # K
heatRate={{HEAT_RATE}}  # K/ps
S0period={{S0_PERIOD}}  # fs
S1therInt={{S1_THER_INT}}  # fs

S1startFrame=$(printf "%08d\n" $S0period)  # fs

# Variables to be computed
elements=($(echo $inpFileName | grep -o "[A-Z][a-z]"))
element1=${elements[0]}
element2=${elements[1]}
potFile=$EAM_DIR/setfl_files/$element1$element2.set
maxJobTime=$(echo "$wallTime - 0.5" | bc)  # hr
timeLimit=$(echo "60*60*$maxJobTime*$numTasks" | bc)  # s
S1period=$(echo "($heatTemp-300)/$heatRate*1000" | bc)  # fs
S1dumpInt=$(echo "$S1period/$totalDumps" | bc)  # fs

mpirun -np $numTasks lmp_openmpi -sf opt \
                                 -in $IN_TEMPLATE \
                                 -var inpFileName $inpFileName \
                                 -var S1startFrame $S1startFrame \
                                 -var element1 $element1 \
                                 -var element2 $element2 \
                                 -var potFile $potFile \
                                 -var timeLimit $timeLimit \
                                 -var initTemp $initTemp \
                                 -var heatTemp $heatTemp \
                                 -var S1period $S1period \
                                 -var S1therInt $S1therInt \
                                 -var S1dumpInt $S1dumpInt \
                          >> ${inpFileName}S$STAGE.log
mkdir ${inpFileName}S$STAGE
mv ${inpFileName}S$STAGE.*.lmp ${inpFileName}S$STAGE