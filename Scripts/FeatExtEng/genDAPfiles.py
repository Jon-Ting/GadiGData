from glob import glob
from multiprocessing import Pool
import os
from os.path import isdir, exists
import pandas as pd
import pickle
import shutil
import tarfile
from zipfile import ZipFile


# Important variables to check!
PROJECT, USER_GROUP_ID, USER_NAME, ELE_COMB = 'q27', '564', 'jt5911', 'AuCo'
sourceDirs = ['L10', 'L12', 'RAL', 'RCS', 'CS']
sourcePaths = [f"/scratch/{PROJECT}/{USER_NAME}/SimAnneal/{ELE_COMB}/{sourceDir}" for sourceDir in sourceDirs]
targetDir = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}"
finalDataPath = f"{targetDir}/finalData"  # Final path to place the results

numFramePerNP, zFillNum = 101, 7
doneFile, outMDfile = 'DONE.txt', 'MDout.csv'
NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
path2NCPacExe = f"/g/data/{PROJECT}/{USER_NAME}/scripts/FeatExtEng/NCPac/{NCPacExeName}"
path2NCPacInp = f"/g/data/{PROJECT}/{USER_NAME}/scripts/FeatExtEng/NCPac/{NCPacInpName}"
headerLine = f"CSIRO Nanostructure Databank - {ELE_COMB} Nanoparticle Data Set"

def renameXYZs():
    print("Copying xyz files to individual directories and relabelling numerically...")
    npCnt = 0  # Different starting point for different type of ordering, e.g. CS != RRAL
    workingList = []
    if not isdir(targetDir): os.mkdir(targetDir)
    for sourcePath in sourcePaths:
        for NPdir in os.listdir(sourcePath):
            print(f"  Nanoparticle: {NPdir}")
            NPdirPath = f"{sourcePath}/{NPdir}"

            # If not done for the nanoparticle yet, reextract Stage 2 files
            if not exists(f"{NPdirPath}/{doneFile}"):
                if not exists(f"{NPdirPath}/{NPdir}S2.zip"):
                    shutil.make_archive(f"{NPdirPath}/{NPdir}S2", 'zip', f"{NPdirPath}", f"{NPdir}S2")
                    print("    Zipped Stage 2 files...")
                with ZipFile(f"{NPdirPath}/{NPdir}S2.zip", 'r') as f: f.extractall(f"{NPdirPath}/")
                print("    Extracted Stage 2 files...")
            else:
                npCnt += 1
                continue
             
            confCnt = 0 + npCnt*numFramePerNP
            for NPconf in os.listdir(f"{NPdirPath}/{NPdir}S2"): 
                if 'min' not in NPconf: continue  # Skip the unminimised configurations
                oriFilePath = f"{NPdirPath}/{NPdir}S2/{NPconf}"
                confID = str(confCnt).zfill(5)
                print(f"  Conformation ID: {confID}")
                confDir = f"{targetDir}/{confID}"
                if not isdir(f"{targetDir}/{confID}"): os.mkdir(confDir)
                
                print("    Copying files...")
                with open(oriFilePath, 'r') as f1:
                    with open(f"{confDir}/{confID}.xyz", 'w') as f2:
                        f2.write(f1.readline())
                        f1.readline()  # Replace second line with CSIRO header
                        f2.write(f"{headerLine} - {NPdir}\n")
                        f2.write(''.join([line for line in f1.readlines()]))
                print("    Replaced header...")
                shutil.copy(path2NCPacExe, f"{confDir}/{NCPacExeName}")
                shutil.copy(path2NCPacInp, f"{confDir}/{NCPacInpName}")
                with open(f"{confDir}/{NCPacInpName}", 'r+') as f:
                    f.writelines([f'{confID}.xyz       - name of xyz input file                                              [in_filexyz]'])
                # os.system(f"head {confDir}/{NCPacInpName}")

                workingList.append((confDir, confID))
                confCnt += 1
            
            # Extract outputs from MD for each configuration
            with open(f"{NPdirPath}/{NPdir}S2.log", 'r') as f1:
                with open(f"{targetDir}/{outMDfile}", 'a') as f2:
                    foundMinLine, prevLine, confCnt = False, None, confCnt - numFramePerNP
                    for (i, line) in enumerate(f1):
                        if '- MINIMISATION -' in line and not foundMinLine: foundMinLine = True
                        elif 'Loop time of' in line and foundMinLine: 
                            confID = str(confCnt).zfill(zFillNum)
                            temp, pres, kinE, potE, totE = prevLine.split()[2:]
                            f2.write(f"{confID},{temp},{pres},{kinE},{potE},{totE}\n")
                            confCnt += 1
                            foundMinLine = False
                        prevLine = line
            assert (confCnt) % numFramePerNP == 0  # Check that confCnt is expected
            npCnt += 1

            # Clean up directory and mark as done
            shutil.rmtree(f"{NPdirPath}/{NPdir}S2")
            open(f"{NPdirPath}/{doneFile}", 'w').close()
            print(f"   {confID} Done!")
    return workingList


def genDAPfiles(work):
    confDir   = work[0]
    confID    = work[1]
    # Execute NCPac.exe for every directories
    os.chdir(confDir)
    os.system(f"./{NCPacExeName}")
    # Move .xyz, FEATURES.csv to {finalDir}
    shutil.copy(f"{confDir}/{confID}.xyz", f"{finalDataPath}/Structures")
    if not exists(f"od_FEATURESET.csv"):  # If execution unsuccessful due to being a BNP instead of TNP
        open(f"{finalDataPath}/Features/{confID}.csv", 'w').close()
    else:
        shutil.copy(f"{confDir}/od_FEATURESET.csv", f"{finalDataPath}/Features/{confID}.csv")
    # Remove unnecessary files
    for f in glob(f"*.mod"): os.remove(f)
    for f in glob(f"fort.*"): os.remove(f)
    for f in glob(f"ov_*"): os.remove(f)
    for f in glob(f"od_*"): 
        if f != 'od_FEATURESET.csv': os.remove(f)
    print(f"   {confID} Done!")


def runNCPacParallel(workingList):
    if not isdir(finalDataPath): os.mkdir(finalDataPath)
    if not isdir(f"{finalDataPath}/Structures"): os.mkdir(f"{finalDataPath}/Structures")
    if not isdir(f"{finalDataPath}/Features"): os.mkdir(f"{finalDataPath}/Features")
    # Check if confDir still exists (could have been run already)
    remainingWork = []
    for workParam in workingList:
        if not isdir(workParam[0]): workingList.remove(workParam)  # If removing confDir after running NCPac
        if not exists(f"{workParam[0]}/od_FEATURESET.csv"): remainingWork.append(workParam)  # If not removing confDir after runnning NCPac (will include BNPs too)
    with Pool() as p: p.map(genDAPfiles, remainingWork)


if __name__ == '__main__':
    workingList = renameXYZs()
    with open('workingList.pickle', 'wb') as f: pickle.dump(workingList, f)
    #with open('workingList.pickle', 'rb') as f: workingList = pickle.load(f)
    #runNCPacParallel(workingList)
    print("All DONE!")

