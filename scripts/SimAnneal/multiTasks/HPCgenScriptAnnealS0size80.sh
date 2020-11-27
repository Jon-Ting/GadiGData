#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=00:20:00,ncpus=1,mem=3GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l wd

bash genScriptAnnealS0size80.sh >> HPCgenScriptAnnealS0size80.log
