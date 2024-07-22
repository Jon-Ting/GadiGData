#!/bin/bash
# Script to check the new output generated from modified codes with known ground truths
# Test case used is OTS1T000

if cmp groundTruths/OTS1T000_surfVoxelBoxCnts.txt groundTruths/OTS1T000_surfVoxelBoxCnts_ori.txt; then echo 'getVoxelBoxCnts() okay!'; fi
head -n -1 groundTruths/OTS1T000_surfExactBoxCntsTimed.txt > groundTruths/OTS1T000_surfExactBoxCnts.txt
if cmp groundTruths/OTS1T000_surfExactBoxCnts.txt groundTruths/OTS1T000_surfExactBoxCnts_ori.txt; then echo 'getSphereBoxCnts() okay!'; fi
