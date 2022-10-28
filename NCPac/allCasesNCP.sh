# Goal: Run NCPac for all test cases
# Author: Jonathan Yik Chang Ting
# Date: 19/5/2021


function rpname {
    echo "Replacing $1 in * with $2"
    for e in *; do mv "$e" "${e/$1/$2}"; done
    echo "Done!"
    }


mkdir featDir/tmp
for testPath in testMolecules/*xyz; do
    testCase=$(echo $testPath | awk -F'/' '{print $NF}')
    echo $testCase
    sed -i 1d NCPac.inp; echo "# Removed first line of NCPac.inp!"
    sed -i "1s/^/${testCase}      - name of xyz input file                                               [in_filexyz] \n/" NCPac.inp; echo "# Added first line of NCPac.inp!"
    ./NCPac.exe; echo "# Ran NCPac.exe!"
    mv od_FEATURESET.csv ov_BOND_* ov_COORD_* ov_Q6Q6.xyz ov_RADIAL_distance.xyz ov_SURF_layer.xyz ov_SURF_classify.xyz featDir/tmp; cd featDir/tmp; echo "# Moved relevant documents into target directory!"
    rpname od ${testCase::-4}_od; rpname ov ${testCase::-4}_ov; mv * ..; cd ../../; echo "# Renamed the relevant documents!"
done
echo -e "\nAll done!"
