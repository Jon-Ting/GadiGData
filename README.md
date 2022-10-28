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
- SimAnneal
    - Raw LAMMPS dataand log files (untracked) generated from simulated annealing of the initial structures along with the input and job script files
- jobLogs (untracked)
    - Log files from various scripts and jobs, organised based on the existing directories in GadiGData directory
- scripts
    - Scripts written for various purposes, organised based on the existing directories in GadiGData directory
- tarFiles (untracked)
    - gunzipped tar ball of various data for backup purposes

Steps to simulate nanoparticles of new morphology:
1. Adjust the parameters in constants.py according to the morphology required
2. Make sure InitStruct/MNP directory exists, otherwise unzip from MDSS
3. Generate the monometallic nanoparticles (MNPs) using genMNP.py 
4. Generate the bimetallic nanoparticles (BNPs) using genBNPCS.py (for core-shell NPs) and genBNPAL.py (for NPs of other ordering)
5. Generate a directory for each new BNPs in simulation directories (located at /scratch) and a config file in it using genBNPdir.sh
6. Generate the LAMMPS input file corresponding to each stage of simulation using genAnnealIn.sh
7. Queue the jobs into the jobList file in /scratch using jobList.sh
8. Submit the jobs to be run using subAnneal.sh


Contents of MDSS system under jt5911/:
- InitStruct.zip
    - Contains initial structures of MNPs and BNPs
- largeBNPs.zip
    - Contains BNPs of diameter > 8 nm
- SimAnneal/
    - minStruct/ (Contains minimised BNP structures, target for ML applications)
        - CSminSmall.zip
        - L10min.zip
        - L12min.zip
        - RALminSmall.zip
        - RCSminSmall.tar.gz
    - oriStruct/ (Contains last frame of BNP structures prior to minimisation)
        - CSoriSmall.zip
        - L10ori.zip
        - L12ori.zip
        - RALoriSmall.zip
        - RCSoriSmall.tar.gz
    - simTraj/ (Contains simulation trajectory documents)
        - CS60+/ (Contains files for BNPs with diameter > 6 nm)
            - Pt80TOPd50ICCS.tar.gz ...
        - CSAu.zip
        - CSCo.zip
        - CSPd.zip
        - CSPt.zip
        - L10All.zip
        - L12All.zip
        - RAL.zip
        - RAL50+/ (Contains files for BNPs with diameter > 5 nm)
            - PtPd80TO50RAL6.tar.gz ...
        - RALAu.zip
        - RALCo.zip
        - RALPd.zip
        - RALPt.zip
        - RCS50+/ (Contains files for BNPS with diameter > 5 nm)
            - PtPd80TO50RCS6.tar.gz ...
        - RCSAu.zip
        - RCSCo.zip
        - RCSPd.zip
        - RCSPt.zip
        - largeCS.zip (Contains files for BNPs with diameter > 8 nm)
        - largeL10.zip (Contains files for BNPs with diameter > 8 nm)
        - largeL12.zip (Contains files for BNPs with diameter > 8 nm) 
        - largeRAL.zip (Contains files for BNPs with diameter > 8 nm) 
        - largeRCS.zip (Contains files for BNPs with diameter > 8 nm) 
