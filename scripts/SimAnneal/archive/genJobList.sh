#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=1,walltime=05:00:00,mem=3GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.7.4

bash jobList.sh > jobList.log
