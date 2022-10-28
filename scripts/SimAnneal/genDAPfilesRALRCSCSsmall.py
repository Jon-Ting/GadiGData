import os
from os.path import isdir, exists
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import shutil
from zipfile import ZipFile


# Important variables to check!
npCnt = 1908
sourceDir = '/g/data/q27/jt5911/PostSim/CSAu'
eleComb = 'AuPd'

# Variables that are more constant
targetDir = f"/scratch/q27/jt5911/SimAnneal/{eleComb}"
numFramePerNP = 101
doneFile = 'DONE.txt'
#NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
#path2NCPacExe = f"/g/data/q27/jt5911/NCPac/{NCPacExeName}"
#path2NCPacInp = f"/g/data/q27/jt5911/NCPac/{NCPacInpName}"

print("Copying xyz files to individual directories and relabelling numerically...")
for NPdir in os.listdir(sourceDir):
    print(f"  Nanoparticle: {NPdir}")
    if eleComb[:2] not in NPdir or eleComb[2:] not in NPdir: continue  # Specify the elemental combination (for CS only)
    #if eleComb not in NPdir: continue  # Specify the elemental combination
    NPdirPath = f"{sourceDir}/{NPdir}"

    # If not done for the nanoparticle yet, reextract Stage 2 files
    if not exists(f"{NPdirPath}/{doneFile}"):
        with ZipFile(f"{NPdirPath}/{NPdir}S2.zip", 'r') as f: f.extractall(f"{NPdirPath}/")
        # print("    Extracted Stage 2 files...")
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
        shutil.copy(oriFilePath, f"{confDir}/{confID}.xyz")
        #shutil.copy(path2NCPacExe, f"{confDir}/{NCPacExeName}")
        #shutil.copy(path2NCPacInp, f"{confDir}/{NCPacInpName}")
        confCnt += 1
    shutil.rmtree(f"{NPdirPath}/{NPdir}S2/")
    open(f"{NPdirPath}/{doneFile}", 'w').close()

    # print("    Done!")
    npCnt += 1

print("All DONE!")




#TODO: Copy files necessary to run NCPac and execute it, merge output files
