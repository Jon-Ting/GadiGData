#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=01:00:00,ncpus=96,mem=30GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l wd

module load python3/3.11.0

#python3 filtRedund.py AuPd20CUL10_665
#python3 filtRedund.py AuPt40CORCS_3871
#python3 filtRedund.py PtAu20THL12_286
#python3 filtRedund.py AuPd30CURAL_1687
#python3 filtRedund.py AuPt40ICRCS_2057
#python3 filtRedund.py AuPd30ICRCS_923
python3 filtRedund.py CoPd40TOL12_4033
