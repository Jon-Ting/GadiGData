#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=1,walltime=05:30:00,mem=5GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

./NCPac.exe > ncp.log
