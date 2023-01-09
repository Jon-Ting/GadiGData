# ASSUMES:
# - targetDir exists, contains 'ConfID, Temp, Pres, KinE, PotE, TotE\n' as header
# - MDout.csv exists in targetDir (contains header)
# TODO: 
# - Copy files necessary to run NCPac and execute it, merge output files
# - Include FD estimated and Q6Q6 distribution averaged in MDout.csv

import os
from os.path import isdir, exists
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import shutil
from zipfile import ZipFile
import tarfile


# Important variables to check!
npCnt = 1866  # Different for RAL and CS
eleComb = 'AuPt'
sourceDir = '/g/data/q27/jt5911/PostSim/L12Au'

# Variables that are more constant
targetDir = f"/scratch/q27/jt5911/SimAnneal/{eleComb}"
numFramePerNP = 101
doneFile = 'DONE.txt'
outMDfile = 'MDout.csv'
#NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
#path2NCPacExe = f"/g/data/q27/jt5911/NCPac/{NCPacExeName}"
#path2NCPacInp = f"/g/data/q27/jt5911/NCPac/{NCPacInpName}"
headerLine = f"CSIRO Nanostructure Databank - {eleComb} Nanoparticle Data Set\n"

print("Copying xyz files to individual directories and relabelling numerically...")
for NPdir in os.listdir(sourceDir):
    print(f"  Nanoparticle: {NPdir}")
    if eleComb[:2] not in NPdir or eleComb[2:] not in NPdir: continue  # Specify the elemental combination
    NPdirPath = f"{sourceDir}/{NPdir}"

    # If not done for the nanoparticle yet, reextract Stage 2 files
    if not exists(f"{NPdirPath}/{doneFile}"):
        # with ZipFile(f"{NPdirPath}/{NPdir}S2.zip", 'r') as f: f.extractall(f"{NPdirPath}/")
        with tarfile.open(f"{NPdirPath}/{NPdir}S2.tar.gz", 'r') as f: f.extractall(f"{NPdirPath}/")  # For L10 and L12 only!
        print("    Extracted Stage 2 files...")
    else:
        # print("    Done!")
        npCnt += 1
        continue

    confCnt = 0 + npCnt*numFramePerNP
    for NPconf in os.listdir(f"{NPdirPath}/{NPdir}S2"): 
        if 'min' not in NPconf: continue  # Skip the unminimised configurations
        oriFilePath = f"{NPdirPath}/{NPdir}S2/{NPconf}"
        confID = str(confCnt).zfill(7)
        print(f"  Conformation ID: {confID}")
        confDir = f"{targetDir}/{confID}"
        if not isdir(f"{targetDir}/{confID}"): os.mkdir(confDir)
        
        # print("    Copying files...")
        with open(oriFilePath, 'r') as f1:
            with open(f"{confDir}/{confID}.xyz", 'w') as f2:
                f2.write(f1.readline())
                f1.readline()  # Replace second line with CSIRO header
                f2.write(headerLine)
                f2.write(''.join([line for line in f1.readlines()]))
        # shutil.copy(oriFilePath, f"{confDir}/{confID}.xyz")
        # shutil.copy(path2NCPacExe, f"{confDir}/{NCPacExeName}")
        # shutil.copy(path2NCPacInp, f"{confDir}/{NCPacInpName}")
        confCnt += 1

    # Extract outputs from MD for each configuration
    with open(f"{NPdirPath}/{NPdir}S2.log", 'r') as f1:
        with open(f"{targetDir}/{outMDfile}", 'a') as f2:
            foundMinLine, prevLine, confCnt = False, None, confCnt - numFramePerNP
            for (i, line) in enumerate(f1):
                if '- MINIMISATION -' in line and not foundMinLine: foundMinLine = True
                elif 'Loop time of' in line and foundMinLine: 
                    confID = str(confCnt).zfill(7)
                    temp, pres, kinE, potE, totE = prevLine.split()[2:]
                    f2.write(f"{confID},{temp},{pres},{kinE},{potE},{totE}\n")
                    confCnt += 1
                    foundMinLine = False
                prevLine = line
    assert (confCnt) % numFramePerNP == 0  # Check that confCnt is expected

    # Clean up directory, mark as done
    shutil.rmtree(f"{NPdirPath}/{NPdir}S2/")
    open(f"{NPdirPath}/{doneFile}", 'w').close()
    print("    Done!")
    npCnt += 1

print("All DONE!")
