#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=06:00:00,ncpus=96,mem=96GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd

# Goal: Script to parallelise running of jobs listed in tarLarge.txt
# Check https://opus.nci.org.au/display/Help/nci-parallel for more info

module load nci-parallel/1.0.0a
export ncores_per_task=1
export ncores_per_numanode=12
 
mpirun -np $((PBS_NCPUS/ncores_per_task)) --map-by ppr:$((ncores_per_numanode/ncores_per_task)):NUMA:PE=${ncores_per_task} nci-parallel --input-file tarLarge.txt --timeout 9000
