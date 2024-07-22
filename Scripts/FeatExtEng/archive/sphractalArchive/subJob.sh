#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l ncpus=12,walltime=03:00:00,mem=160GB,jobfs=80GB
#PBS -l storage=scratch/vp91
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.10.4
module load nvidia-hpc-sdk/22.11

#PBS -l ncpus=1,walltime=03:00:00,mem=30GB,jobfs=10GB
#nsys profile --trace=nvtx,osrt --force-overwrite=true --stats=true --output=profileOutputs/findAtomNeighs python3 numbaExp.py

nsys profile --trace=nvtx,osrt --force-overwrite=true --stats=true --output=profileOutputs/exNoConcOri python3 runCase.py
# --nvtx-domain-exclude/include=XXX,XXX,XXX
