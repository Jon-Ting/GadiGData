#!/bin/bash
#PBS -P q27
#PBS -q hugemem
#PBS -l ncpus=2,walltime=15:00:00,mem=1500GB,jobfs=1000GB
#PBS -l storage=gdata/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

./NCPac.exe
