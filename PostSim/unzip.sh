#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=10:00:00,ncpus=1,mem=10GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l other=mdss
#PBS -l wd

# Goal: Script to move backed-up files from massdata/ to gdata/
# Author: Jonathan Yik Chang Ting
# Date: 6/5/2022

#unzip RALAu.zip
unzip CSAu.zip
