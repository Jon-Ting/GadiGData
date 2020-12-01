#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=00:15:00,ncpus=1,mem=2GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l wd

bash genScriptAnnealS0size20.sh >> HPCgenScriptAnnealS0size20.log
