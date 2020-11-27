# README file for GadiGData directory
# Author: Jonathan Yik Chang Ting
# Date: 26/11/2020

A repository acting as a backup for files generated throughout the author's PhD program that are not raw-data on NCI HPC cluster Gadi.
The files contained herein could be similar to the ones in the Workstation repository as most of the codes were first developed in the author's local machine which is synced to the aforementioned repository and transferred to Gadi for deployment.
The type of information stored in each directory is as below:

- Benchmark:
    - LAMMPS input and potential files used for the benchmarking simulation runs
    - All raw LAMMPS data files from the simulations run (untracked)
    - A gunzipped tar ball containing all of the simulations performed for benchmarking purposes
- EAM
    - setfl directory contains the generated EAM alloy potentials
- InitStruct (untracked)
    - Initial structures for MD simulations (monometallic, bimetallic) in LAMMPS data file format
- SimAnneal (untracked)
    - Raw LAMMPS data files generated from simulated annealing of the initial structures along with the input and job script files
- jobLogs
    - Log files from various scripts and jobs, organised based on the existing directories in GadiGData directory
- scripts
    - Scripts written for various purposes, organised based on the existing directories in GadiGData directory
- tarFiles (untracked)
    - gunzipped tar ball of various data for backup purposes
