#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=03:00:00,ncpus=144,mem=150GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

module load lammps/3Mar2020

mpirun -n 144 lmp_openmpi -in PtAu150COL12S1.in > PtAu150COL12S1.log
mkdir PtAu150COL12S1
mv PtAu150COL12S1.*.lmp PtAu150COL12S1
tar -czvf PtAu150COL12S1.tar.gz PtAu150COL12S1
rm -rf PtAu150COL12S1
