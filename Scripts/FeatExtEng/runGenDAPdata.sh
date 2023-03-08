#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=192,walltime=30:00:00,mem=80GB,jobfs=0GB
#PBS -l storage=gdata/q27+scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.11.0

python3 genCSVs.py  
#python3 mergeFeatures.py 

# Resources 
## Small NPs
#PBS -l ncpus=1,walltime=03:00:00,mem=50GB,jobfs=0GB  # setupNCPac
#PBS -l ncpus=96,walltime=70:00:00,mem=10GB,jobfs=0GB  # filtRedund
#PBS -l ncpus=96,walltime=36:00:00,mem=50GB,jobfs=0GB  # runNCPac
#PBS -l ncpus=96,walltime=01:00:00,mem=50GB,jobfs=0GB  # mergeReformatData
#PBS -l ncpus=1,walltime=01:30:00,mem=50GB,jobfs=0GB  # concatNPfeats
#PBS -l ncpus=1,walltime=02:00:00,mem=40GB,jobfs=0GB  # reorderIdxs
## Large NPs
#PBS -l ncpus=1,walltime=18:00:00,mem=40GB,jobfs=0GB
#PBS -l ncpus=48,walltime=36:00:00,mem=110GB,jobfs=0GB
#PBS -l ncpus=48,walltime=48:00:00,mem=100GB,jobfs=0GB
#PBS -l ncpus=1,walltime=10:00:00,mem=100GB,jobfs=0GB
