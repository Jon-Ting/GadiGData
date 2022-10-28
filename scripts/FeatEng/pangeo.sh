#!/bin/bash
#PBS -N pangeo_test
#PBS -P q27
#PBS -q normal
#PBS -l walltime=05:00:00,ncpus=1,mem=10GB,jobfs=10GB,storage=scratch/q27+gdata/q27

module load pangeo/2021.01
pangeo.ini.all.sh
sleep infinity
