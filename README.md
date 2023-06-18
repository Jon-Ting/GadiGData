# BNPcaptaGen
This repository contains code written for generation of bimetallic nanoparticles (BNPs) structural data set for machine learning applications.

Conducted by: Jonathan Yik Chang Ting
Supervised by: Amanda Barnard
Institution: School of Computing Australian National University
Program: Doctor of Philosophy in ANU College of Engineering and Computer Science
Expected Finishing Date Accomplished: 6/4/24

A repository acting as a backup for files generated throughout the author's PhD program that are not raw-data on NCI HPC cluster Gadi.
The files contained herein could be similar to the ones in the Workstation repository as most of the codes were first developed in the author's local machine which is synced to the aforementioned repository and transferred to Gadi for deployment.
The type of information stored in each directory is as below:

## Contents of each directory
- Benchmark:
    - LAMMPS input and potential files used for the benchmarking simulation runs
    - All raw LAMMPS data files from the simulations run (untracked)
    - A gunzipped tar ball containing all of the simulations performed for benchmarking purposes
- EAM
    - Modified LAMMPS tools for the generation of relevant interatomic potential files required for LAMMPS scripts execution
    - setfl directory containing the generated EAM binary alloy potentials
- InitStruct (untracked)
    - Initial structures for MD simulations (monometallic, bimetallic) in LAMMPS data file format
- PostSim
    - Raw LAMMPS data (untracked) generated from simulated annealing of the initial structures along with the input and job script files
- scripts
    - Scripts written and log files (untracked) for various purposes #TODO: Elaborate!

## Instructions to use the repository to generate more BNPs structural data
- The current script is designed to:
    - only generate BNPs with different combinations of the listed degrees of freedom, but extension is possible by appropriate modification of the code.
    - be run on high performance computing cluster such as Gadi of National Computational Infrastructure.

## Degrees of freedom of BNPs generated
- Elemental composition: Au, Pt, Pd, Co
- Size: 20, 30, 40, 50, 60, 70, 80 Angstroms
- Shape: 
    - cuboctahedron (CO)
    - cube (CU)
    - decahedron (DH)
    - icosahedron (IC)
    - octahedron (OT)
    - rhombic dodecahedron (RD)
    - tetrahedron (TH)
    - truncated octahedron (TO)
- Ratio: i:j where i, j are from {25, 50, 75}, with an additional constraint if i+j == 100
- Atomic ordering: 
    - CS: core-shell
    - RCS: core-shell with reduced probabililty of core element towards the shell
    - RAL: random solid solution
    - L10: L10 intermetallic compound
    - L12: L12 intermetallic compound

### Generation of BNP initial structures
1. Adjust the parameters in constants.py according to the morphology required
2. Generate the monometallic nanoparticles (MNPs) using genMNP.py 
3. Generate the bimetallic nanoparticles (BNPs) using genBNPCS.py (for core-shell NPs) and genBNPAL.py (for NPs of other ordering)

### Simulation of BNPs
#### Stages of simulations
- S0: Short equilibration of TNPs
- S1: Heating up of TNPs, saving configurations along the way
- S2: Short equilibration of the saved TNP configurations at the saved temperature

#### Instructions:
1. Generate a directory for each new BNPs in simulation directories (located at /scratch) and a config file in it using genBNPdir.sh
2. Generate the LAMMPS input file corresponding to each stage of simulation using genAnnealIn.sh
3. Queue the jobs into the jobList file in /scratch using jobList.sh
4. Submit the jobs to be run using subAnneal.sh

### Feature extraction of BNPs
1. Go to ./FeatExtEng/
2. Modify the paths and parameters in genCSVs.py as appropriate.
3. Submit runGenDAPdata.sh to the HPC. This will generate:
    - {MDout.csv}, which contains the output of MD simulations of all BNPs.
    - {features.csv}, which contains the features extracted by NCPac for all BNPs.
4. Modify the parameters in mergeFeatures.py and run it. This will merge the information from the 2 csv files and generate a new {*_nanoparticle_dataset.csv} following the format of dataset stored on CSIRO's Data Access Portal, such as https://data.csiro.au/collection/csiro%3A58177v1 (AuPt nanoparticle). 

## Contents of MDSS system under jt5911/:
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
