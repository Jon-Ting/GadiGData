#!/bin/bash
#PBS -P q27
#PBS -q copyq
#PBS -l ncpus=1,walltime=10:00:00,mem=192GB,jobfs=200GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

nci-file-expiry batch-recover final_list_of_files.txt
