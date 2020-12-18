#!/bin/bash
#PBS -P q27
#PBS -q hugemem
#PBS -l walltime=24:00:00,ncpus=1,mem=500GB,jobfs=1GB
#PBS -l wd

tar -czvf largeBNPs.tar.gz largeBNPs/ > zipLargeBNPs.log
