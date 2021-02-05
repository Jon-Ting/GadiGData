#!/bin/bash
# Goal: Script to finish the post execution tasks of unfinished jobs
# Author: Jonathan Yik Chang Ting
# Date: 10/1/2021

errFileDir=/scratch/q27/jt5911/SimAnneal/err
for stdOutFile in $(ls *.sh.o*); do
    jobName=$(head -1 $stdOutFile | awk '{print $2}'); dirName=${jobName%/*}
    if grep -q "re" <<< "${jobName: -2}"; then initName=${jobName::-2}; else initName=$jobName; fi
    unqName=$(echo $initName | awk -F'/' '{print $NF}')
    echo $dirName; cd $dirName;
    mkdir $unqName; mv scratch/*/*/*/*/*/*/*lmp $unqName; mv *lmp $unqName; tar -czf $unqName.tar.gz $unqName; rm -rf $unqName scratch
    cd $errFileDir
done
