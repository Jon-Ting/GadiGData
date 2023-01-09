#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=1,walltime=24:00:00,mem=50GB,jobfs=1GB
#PBS -l storage=scratch/q27+gdata/q27
#PBS -l wd

# Goal: Extract features from nanoparticle structures using NCPac
# Author: Jonathan Yik Chang Ting
# Date: 2/2/2021
# TO DO
# - 


NCP_PATH=/g/data/$PROJECT/$USER/NCPac
DATA_DIR=/scratch/$PROJECT/$USER/SimAnneal/AuPt
declare -a TYPE_ARR=('L10' 'L12' 'CS' 'RCS' 'RAL')
SURF_FILE=surf.xyz


# Loop through all NP simulation directories
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    # Go to the directory where the raw data are stored
    cd $DATA_DIR/${bnpType}min
    # Loop through all NP in the directory
    for filePath in $DATA_DIR/${bnpType}min/*; do
        xyzName=$(echo $filePath | awk -F'/' '{print $NF}')
        unqName=${xyzName::-4}
        # Identify constituent elements and compute appropriate first neighbour cutoff
        ele1=${xyzName:0:2}; ele2=${xyzName:2:2}
        if [ $ele1 == 'Au' ]; then cut1=3.6
        elif [ $ele1 == 'Co' ]; then cut1=3.4
        elif [ $ele1 == 'Pd' ]; then cut1=3.5
        elif [ $ele1 == 'Pt' ]; then cut1=3.7
        else echo 'Element 1 unrecognised'; fi 
        if [ $ele2 == 'Au' ]; then cut2=3.6
        elif [ $ele2 == 'Co' ]; then cut2=3.4
        elif [ $ele2 == 'Pd' ]; then cut2=3.5
        elif [ $ele2 == 'Pt' ]; then cut2=3.7
        else echo 'Element 1 unrecognised'; fi 
        cut12=$(echo "($cut1 + $cut2) / 2" | bc)
        echo $unqName $ele1 $ele2 $cut1 $cut2 $cut12
        # Copy NCPac executable and input files to the directory
         cp -r $NCP_PATH/NCPac.exe $NCP_PATH/NCPac.inp ./
        # Modify NCPac.inp (Need further modification)
        sed -i "s/^.*in_filexyz.*$/$xyzName        - name of xyz input file        [in_filexyz]/" NCPac.inp
        sed -i "s/^.*in_cutoff.*$/$ele1 $cut1 $cut12                                  - NN unique cutoff matrix (line1 type1,r1r1,r1r2, line2 type2 r2r2)   [in_cutoff(i,j)]/" NCPac.inp
        sed -i "s/^.*Second element cutoff.*$/$ele2 $cut2                                      - Second element cutoff/" NCPac.inp
        # Run NCPac.exe to get od_FEATURESET.csv and od_LINDEX.dat
        ./NCPac.exe >> ncp.log
        # Rename the essential output files (could append a lot more files to rename)
        mv od_FEATURESET.csv od_FEATURESET_$unqName.csv
        #mv od_LINDEX.dat od_LINDEX_$unqName.dat

        # Modify NCPac.inp to get surface-dependent g(r) and off Lindemann index calculation
        #mv ov_SURF_layer.xyz $SURF_FILE
        #sed -i "s/^.*in_filexyz.*$/$SURF_FILE        - name of xyz input file        [in_filexyz]/" NCPac.inp
        #sed -i "s/^.*in_5th_option.*$/1        - read in surface data 5th column        (0=N,1=Y)  [in_5th_option]/" NCPac.inp
        #sed -i "s/^.*in_lindem_flag.*$/0        - LINDEMANN INDEX        (0=N,1=Y)  [in_lindem_flag]" NCPac.inp
        # Rerun NCPac.exe
        #./NCPac.exe
    done
done
