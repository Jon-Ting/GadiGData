#!/bin/bash
# Goal: Extract features from nanoparticle structures using NCPac
# Author: Jonathan Yik Chang Ting
# Date: 2/2/2021
# To do:


NCPPath=/g/data/$PROJECT/$USER/NCPac
JOB_LIST_FILE=jobList; QUEUE_LIST_FILE=queueList
CONFIG_FILE=config.yml; RUN_LOCK=run.lock; EXAM_LOCK=examine.lock; SURF_FILE=surf.xyz
SIM_DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal
# Loop through all NP simulation directories
for dirPath in $SIM_DATA_DIR/*/*; do
    if grep -Fq $dirPath $SIM_DATA_DIR/$JOB_LIST_FILE; then echo "$dirPath on list, skipping..."; continue  # On the to-be-submitted-list
    elif grep -Fq $dirPath $SIM_DATA_DIR/$QUEUE_LIST_FILE; then echo "$dirPath queuing, skipping..."; continue  # Has been submitted
    elif test -f $dirPath/$RUN_LOCK; then echo "$dirPath running a job, skipping..."; continue  # Running
    fi
    # Check that S2 is done
    runState=$(grep S2ok: $dirPath/$CONFIG_FILE); dirName=$(echo $dirPath | awk -F'/' '{print $NF}')
    if grep -q "true" <<< "$runState" ; then echo "$dirName S2 OK!"; else echo "$dirName S2 not OK"; continue; fi
    # Change into the directry, untar the S2 tar.gz file; change into S2 directory, copy NCPac files to the directory
    cd $dirPath; tar -xf *S2.tar.gz; cd *S2; cp -r $NCPPath/NCPac.exe $NCPPath/NCPac.inp $dirPath/*S2/
    # Loop through all NP in the S2 directory
    for filePath in *; do
        # Modify NCPac.inp to run on the file 
        sed -i "s/^.*in_filexyz.*$/$filePath        - name of xyz input file        [in_filexyz]/" NCPac.inp
        # Run NCPac.exe to get od_FEATURESET.csv and od_LINDEX.dat
        ./NCPac.exe
        # Rename the essential output files
        mv od_FEATURESET.csv od_FEATURESET_whole.csv; mv od_LINDEX.dat od_LINDEX_whole.dat
        # Modify NCPac.inp to get surface-dependent g(r) and off Lindemann index calculation
        mv ov_SURF_layer.xyz $SURF_FILE
        sed -i "s/^.*in_filexyz.*$/$SURF_FILE        - name of xyz input file        [in_filexyz]/" NCPac.inp
        sed -i "s/^.*in_5th_option.*$/1        - read in surface data 5th column        (0=N,1=Y)  [in_5th_option]/" NCPac.inp
        sed -i "s/^.*in_lindem_flag.*$/0        - LINDEMANN INDEX        (0=N,1=Y)  [in_lindem_flag]" NCPac.inp
        # Rerun NCPac.exe
        ./NCPac.exe
        # Collect the data into a table

    done
    # Remove the executable and redundant files, recompress the S2 directory, remove S2 directory
    rm -f NCPac.exe NCPac.inp od* ov*; cd ..; tar -czf $dirName.tar.gz $dirName; rm -rf $dirName
done
