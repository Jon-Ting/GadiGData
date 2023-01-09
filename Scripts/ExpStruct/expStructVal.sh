# Goal: Validate the experimental structures generated using standard RMC provided by collaborators
# Author: Jonathan Yik Chang Ting
# Date: 9/2/2021


archiveDir=/g/data/q27/jt5911/scripts/SimAnneal/archive
expStructDir=/g/data/q27/jt5911/ExpStruct
NCPDir=/g/data/q27/jt5911/NCPac
cd $NCPDir
for expStructFile in $expStructDir/*; do
    fileName=$(echo $expStructFile | awk -F'/' '{print $NF}'); echo $fileName; cp $expStructFile .
    sed -i "s|^.*name of xyz input file.*$|$fileName                - name of xyz input file        [in_filexyz]|" NCPac.inp
    ./NCPac.exe
    mv od_BOND_length.csv ${expStructFile::-4}_od_BOND_length.csv
    mv od_FEATURESET.csv ${expStructFile::-4}_od_FEATURESET.csv
    mv od_G3.csv ${expStructFile::-4}_od_G3.csv
    mv od_GR.csv ${expStructFile::-4}_od_GR.csv
    mv od_Q6Q6.csv ${expStructFile::-4}_od_Q6Q6.csv
    mv ov_CLUSTER_filtered.xyz ${expStructFile::-4}_ov_CLUSTER_filtered.xyz
    mv ov_BOND_length.xyz ${expStructFile::-4}_ov_BOND_length.xyz
    mv ov_Q6Q6.xyz ${expStructFile::-4}_ov_Q6Q6.xyz
    sed -i 
    lmp_openmpi -in $archiveDir/SPE.in
done
echo "Done!"
