#!/bin/bash
# Goal: Determine whether NP has melted via Lindemann index and PE plots
# Author: Jonathan Yik Chang Ting
# Date: 6/1/2021

SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
SCRIPT_DIR=/g/data/$PROJECT/$USER/scripts/SimAnneal
NCPAC_DIR=/home/564/$USER/NCPac

# Convert LAMMPS dump files into xyz format
bnpPath=$SIM_DATA_DIR/L10/AuCo20COL10
dumpDirPath=$bnpPath//AuCo20COL10S1
pizza.py -x gl -f $SCRIPT_DIR/dump2xyz.py $dumpDirPath

# Compute Lindemann index using NCPac
$NCP
mv $NP/od_LINDEX.dat $bnpPath/

# Determine the melting point (if melted)
threshold=0.1
# awk to get columns
# conditions if index transitions from ordered to disordered, break from loop when achieved
