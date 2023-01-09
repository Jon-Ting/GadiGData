- MDout.csv might contain duplicates, vim ':sort u' to leave only unique lines and sort them
* Last frames and minimised structures NPs in checkHere have been extracted
NPs in toRerun.txt potentially need to undergo another round of Stage 2 simulations

npCnt (for each $eleComb):
- 0 for RCSsmall
- 720 for RALsmall
- 1440 for CSsmall
- 1782 for L10
- 1824 for L12
- 1908 for large RCS
- 2868 for large RAL
- 3828 for CS
- Total 4788

- Procedure:
    - Get .zip from MDSS (getZip.sh)
    - Unzip them (unzip.sh)
    - Make prerequisite directories and files (mkdir {eleComb}, touch {MDout.csv})
    - Generate DAP files (genDAP*.sh)
    - Check non-empty before removing source files
