#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime={{WALLTIME}}:00:00,ncpus={{NUM_TASKS}},mem={{MEM}}GB,jobfs={{JOBFS}}GB,software=lammps
#PBS -l storage=gdata/q27
#PBS -l wd

module load lammps/3Mar2020

# Fixed variables
STAGE=3
GDATA_DIR=/g/data/q27/jt5911
EAM_DIR=$GDATA_DIR/EAM

# Variables to be substituted
wallTime={{WALLTIME}}  # hr
numTasks={{NUM_TASKS}}
totalDumps={{TOTAL_DUMPS}}
inpFileName={{INP_FILE_NAME}}
heatTemp={{HEAT_TEMP}}
coolTemp={{COOL_TEMP}}
coolRate={{COOL_RATE}}  # K/ps
S2period={{S2_PERIOD}}  # fs, total of 1 ns
S3therInt={{S3_THER_INT}}  # fs

S3startFrame=$(printf "%08d\n" $S2period)  # fs

# Variables to be computed
elements=($(echo $inpFileName | grep -o "[A-Z][a-z]"))
element1=${elements[0]}
element2=${elements[1]}
potFile=$EAM_DIR/setfl_files/$element1$element2.set
maxJobTime=$(echo "$wallTime - 0.5" | bc)  # hr
timeLimit=$(echo "60*60*$maxJobTime*$numTasks" | bc)  # s
S3period=$(echo "($heatTemp-300)/$coolRate*1000" | bc)  # fs
S3dumpInt=$(echo "$S3period/$totalDumps" | bc)  # fs

mpirun -np $numTasks lmp_openmpi -sf opt \
                                 -in ${inpFileName}S$STAGE.in \
                                 -var inpFileName $inpFileName \
                                 -var S3startFrame $S3startFrame \
                                 -var element1 $element1 \
                                 -var element2 $element2 \
                                 -var potFile $potFile \
                                 -var timeLimit $timeLimit \
                                 -var heatTemp $heatTemp \
                                 -var coolTemp $coolTemp \
                                 -var S3period $S3period \
                                 -var S3therInt $S3therInt \
                                 -var S3dumpInt $S3dumpInt \
                                 >> ${inpFileName}S$STAGE.log
mkdir ${inpFileName}S$STAGE
mv ${inpFileName}S$STAGE.*.lmp ${inpFileName}S$STAGE
tar -czvf ${inpFileName}S$STAGE.tar.gz ${inpFileName}S$STAGE
rm -rf ${inpFileName}S$STAGE
