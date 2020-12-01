# Script to automate the generation of LAMMPS simulations job scripts for each nanoparticle to be simulated
# Author: Jonathan Yik Chang Ting
# Date: 24/11/2020
# To do:

declare -a TYPE_ARR=('CS/' 'L10/' 'L12/' 'RCS/' 'RAL/')
TARGET_SIZE=20
STAGE=0

wallTime=$(printf "%02d\n" 3)  # hr
numTasks=24
mem=4  # GB
jobFS=1  # GB
totalDumps=100
initTemp=300  # K
heatTemp=1300  # K
coolTemp=300  # K
heatRate=0.25  # K/ps
coolRate=0.15  # K/ps
S0period=500000  # fs
S2period=1000000  # fs
S0therInt=100  # fs
S1therInt=500  # fs
S2therInt=200  # fs
S3therInt=500  # fs

SIM_DATA_DIR=/scratch/q27/jt5911
GDATA_DIR=/g/data/q27/jt5911
EAM_DIR=$GDATA_DIR/EAM
TEMPLATE_NAME=$GDATA_DIR/scripts/SimAnneal/annealS$STAGE

echo "Looping through directories:"
echo "-----------------------------------------------"
for ((i=0;i<${#TYPE_ARR[@]};i++)); do
    bnpType=${TYPE_ARR[$i]}
    inpDirName=BNP/$bnpType
    annealDirName=$SIM_DATA_DIR/SimAnneal/$bnpType
    echo "  $bnpType Directory:"
    for bnpDir in $annealDirName*; do
        
        # Identify the targeted directories
        inpFileName=$(echo $bnpDir | grep -oP "(?<=$bnpType).*")
        if [ $bnpType == 'CS/' ]; then  # CS: Au150THPd40COCS
            if [[ $inpFileName =~ ^[A-Z][a-z]$TARGET_SIZE[A-Z]{2}[A-Z][a-z][0-9]{2,}[A-Z]{4,}$ ]]; then true; else continue; fi
        elif [ $bnpType == 'L10/' ] || [ $bnpType == 'L12/' ]; then  # L10, L12: CoAu150COL10, CoPd150TOL12
            if [[ $inpFileName =~ ^([A-Z][a-z]){2}$TARGET_SIZE[A-Z]{2}L1(0|2)$ ]]; then true; else continue; fi
        elif [ $bnpType == 'RAL/' ] || [ $bnpType == 'RCS/' ]; then  # RAL, RCS: CoAu150TO25RAL4, CoPd150TO75RCS6
            if [[ $inpFileName =~ ^([A-Z][a-z]){2}$TARGET_SIZE[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]$ ]]; then true; else continue; fi
        fi
        echo "    $inpFileName"

        # Skip if file has been generated
        # if test -f ${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.sh; then
        #     echo "      Job script generated! Skipping..."
        #     continue
        # fi
        
        # Copy the template files to target directory
        cp $TEMPLATE_NAME.in ${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.in
        cp $TEMPLATE_NAME.sh ${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.sh
        JOB_SCRIPT=${annealDirName}/${inpFileName}/${inpFileName}S$STAGE.sh
        echo "      Scripts copied!"
        
        # Substitute variables in template job script files
        sed -i "s/{{WALLTIME}}/$wallTime/g" $JOB_SCRIPT
        sed -i "s/{{NUM_TASKS}}/$numTasks/g" $JOB_SCRIPT
        sed -i "s/{{MEM}}/$mem/g" $JOB_SCRIPT
        sed -i "s/{{JOBFS}}/$jobFS/g" $JOB_SCRIPT
        sed -i "s/{{TOTAL_DUMPS}}/$totalDumps/g" $JOB_SCRIPT
        sed -i "s/{{INP_FILE_NAME}}/$inpFileName/g" $JOB_SCRIPT
        if [ $STAGE -eq 0 ]; then
            sed -i "s|{{INP_DIR_NAME}}|$inpDirName|g" $JOB_SCRIPT
            sed -i "s/{{INIT_TEMP}}/$initTemp/g" $JOB_SCRIPT
            sed -i "s/{{S0_PERIOD}}/$S0period/g" $JOB_SCRIPT
            sed -i "s/{{S0_THER_INT}}/$S0therInt/g" $JOB_SCRIPT
        elif [ $STAGE -eq 1 ]; then
            sed -i "s/{{INIT_TEMP}}/$initTemp/g" $JOB_SCRIPT
            sed -i "s/{{HEAT_TEMP}}/$heatTemp/g" $JOB_SCRIPT
            sed -i "s/{{HEAT_RATE}}/$heatRate/g" $JOB_SCRIPT
            sed -i "s/{{S0_PERIOD}}/$S0period/g" $JOB_SCRIPT
            sed -i "s/{{S1_THER_INT}}/$S1therInt/g" $JOB_SCRIPT
        elif [ $STAGE -eq 2 ]; then
            sed -i "s/{{INIT_TEMP}}/$initTemp/g" $JOB_SCRIPT
            sed -i "s/{{HEAT_TEMP}}/$heatTemp/g" $JOB_SCRIPT
            sed -i "s/{{HEAT_RATE}}/$heatRate/g" $JOB_SCRIPT
            sed -i "s/{{S2_PERIOD}}/$S2period/g" $JOB_SCRIPT
            sed -i "s/{{S2_THER_INT}}/$S2therInt/g" $JOB_SCRIPT 
        else
            sed -i "s/{{HEAT_TEMP}}/$heatTemp/g" $JOB_SCRIPT
            sed -i "s/{{COOL_TEMP}}/$coolTemp/g" $JOB_SCRIPT
            sed -i "s/{{COOL_RATE}}/$coolRate/g" $JOB_SCRIPT
            sed -i "s/{{S2_PERIOD}}/$S2period/g" $JOB_SCRIPT
            sed -i "s/{{S3_THER_INT}}/$S3therInt/g" $JOB_SCRIPT 
        fi
        echo "      Variables substituted!"
    done
done
echo -e "Done!\n"

