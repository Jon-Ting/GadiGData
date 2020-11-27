#!/bin/bash
#PBS -P q27
#PBS -q express
#PBS -l walltime=00:20:00,ncpus=1,mem=2GB,jobfs=1GB
#PBS -l storage=gdata/q27
#PBS -l wd

bash genScriptAnnealS0size40.sh >> HPCgenScriptAnnealS0size40.log
