#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=1,walltime=05:00:00,mem=8GB,storage=scratch/q27+gdata/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.8.5
python3 pltLogProp.py >> eq.log
sed -i "0,/^.*dirNameStr =.*$/s//        dirNameStr = 'L12'/" pltLogProp.py
python3 pltLogProp.py >> eq.log
