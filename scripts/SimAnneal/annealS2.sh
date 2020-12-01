#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime={{WALLTIME}}:00:00,ncpus={{NUM_TASKS}},mem={{MEM}}GB,jobfs={{JOBFS}}GB,software=lammps
#PBS -l storage=gdata/q27
#PBS -l wd

module load lammps/3Mar2020

# Fixed variables
STAGE=2
GDATA_DIR=/g/data/q27/jt5911
EAM_DIR=$GDATA_DIR/EAM

# Variables to be substituted
wallTime={{WALLTIME}}  # hr
numTasks={{NUM_TASKS}}
totalDumps={{TOTAL_DUMPS}}
inpFileName={{INP_FILE_NAME}}
initTemp={{INIT_TEMP}}  # K
heatTemp={{HEAT_TEMP}}
heatRate={{HEAT_RATE}}  # K/ps
S2period={{S2_PERIOD}}  # fs, total of 1 ns
S2therInt={{S2_THER_INT}}  # fs

S2startFrame=$(printf "%08d\n" $S1period)  # fs

# Variables to be computed
elements=($(echo $inpFileName | grep -o "[A-Z][a-z]"))
element1=${elements[0]}
element2=${elements[1]}
potFile=$EAM_DIR/setfl_files/$element1$element2.set
maxJobTime=$(echo "$wallTime - 0.5" | bc)  # hr
timeLimit=$(echo "60*60*$maxJobTime*$numTasks" | bc)  # s
S1period=$(echo "($heatTemp-300)/$heatRate*1000" | bc)  # fs
S2dumpInt=$(echo "$S2period/$totalDumps" | bc)  # fs

mpirun -np $numTasks lmp_openmpi -sf opt \
                                 -in ${inpFileName}S$STAGE.in \
                                 -var inpFileName $inpFileName \
                                 -var S2startFrame $S2startFrame \
                                 -var element1 $element1 \
                                 -var element2 $element2 \
                                 -var potFile $potFile \
                                 -var timeLimit $timeLimit \
                                 -var heatTemp $heatTemp \
                                 -var S2period $S2period \
                                 -var S2therInt $S2therInt \
                                 -var S2dumpInt $S2dumpInt \
                                 >> ${inpFileName}S$STAGE.log
mkdir ${inpFileName}S$STAGE
mv ${inpFileName}S$STAGE.*.lmp ${inpFileName}S$STAGE
tar -czvf ${inpFileName}S$STAGE.tar.gz ${inpFileName}S$STAGE
rm -rf ${inpFileName}S$STAGE
