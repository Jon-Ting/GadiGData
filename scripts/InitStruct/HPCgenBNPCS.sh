#!/bin/bash
#PBS -P q27
#PBS -q hugemem
#PBS -l walltime=24:00:00,ncpus=48,mem=500GB,jobfs=1GB,software=python+lammps
#PBS -l storage=gdata/q27
#PBS -l wd

module load python3/3.8.5
module load lammps/3Mar2020

bash genBNPCS.sh >> HPCgenBNPCS.log
