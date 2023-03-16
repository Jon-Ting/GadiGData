from glob import glob
from multiprocessing import Pool
import os
from os.path import isdir, exists
import pandas as pd
import pickle
import re
import shutil
from zipfile import ZipFile


runTask = 'runNCPac'  # 'setupNCPac' or 'runNCPac' 
runParallel = True
    
PROJECT, USER_GROUP_ID, USER_NAME, ELE_COMB = 'q27', '564', 'jt5911', 'AuPdPt'
sourceDirs = ['CS', 'LL10', 'CL10S', 'CRALS', 'CSL10', 'L10R', 'CSRAL', 'RRAL', 'CRSR']
sourcePaths = [f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}_MDsim/{directory}" for directory in sourceDirs]
targetDir = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}"
finalDataPath = f"{targetDir}/finalData"
numFramePerNP = 11
zFillNum = 5
doneFile = 'DONE.txt'
outMDfile = 'MDout.csv'
NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
path2NCPacExe = f"/home/{USER_GROUP_ID}/{USER_NAME}/TNPgeneration/FeatExtEng/NCPac/{NCPacExeName}"
path2NCPacInp = f"/home/{USER_GROUP_ID}/{USER_NAME}/TNPgeneration/FeatExtEng/NCPac/{NCPacInpName}"
headerLine = f"CSIRO Nanostructure Databank - {ELE_COMB} Nanoparticle Data Set"

def setupNCPac():
    print("Copying xyz files to individual directories and relabelling numerically...")
    npCnt = 0
    workingList = []
    if not isdir(targetDir): os.mkdir(targetDir)
    with open(f"{targetDir}/{outMDfile}", 'a') as f: f.write('confID,T,P,PE,KE,TE\n')
    for sourcePath in sourcePaths:
        for NPdir in os.listdir(sourcePath):
            print(f"  Nanoparticle: {NPdir}")
            NPdirPath = f"{sourcePath}/{NPdir}"

            # If not done for the nanoparticle yet, reextract Stage 2 files
            if not exists(f"{NPdirPath}/{doneFile}"):
                with ZipFile(f"{NPdirPath}/{NPdir}S2.zip", 'r') as f: f.extractall(f"{NPdirPath}/")
                print("    Extracted Stage 2 files...")
            else:
                npCnt += 1
                continue
             
            confCnt = npCnt*numFramePerNP
            allS2NPs = [np for np in os.listdir(f"{NPdirPath}/{NPdir}S2") if 'min' in np]
            for NPconf in sorted(allS2NPs, key=lambda key: [int(i) for i in re.findall('min.([0-9]+)', key)]):
                oriFilePath = f"{NPdirPath}/{NPdir}S2/{NPconf}"
                confID = str(confCnt).zfill(zFillNum)
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
                    f.writelines([f"{confID}.xyz       - name of xyz input file                                              [in_filexyz]"])

                workingList.append((confDir, confID))
                confCnt += 1
            
            # Extract outputs from MD for each configuration
            with open(f"{NPdirPath}/{NPdir}S2.log", 'r') as f1:
                with open(f"{targetDir}/{outMDfile}", 'a') as f2:
                    foundMinLine, prevLine, confCnt = False, None, confCnt - numFramePerNP
                    for line in f1:
                        if '- MINIMISATION -' in line and not foundMinLine: foundMinLine = True
                        elif 'Loop time of' in line and foundMinLine: 
                            confID = str(confCnt).zfill(zFillNum)
                            temp, pres, potE, kinE, totE = prevLine.split()[2:]
                            f2.write(f"{confID},{temp},{pres},{potE},{kinE},{totE}\n")
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


def runNCPac(work, verbose=False):
    confDir, confID = work
    # Execute NCPac.exe for every directories
    if verbose: print(f"    Running NCPac for {confID}...")
    os.chdir(confDir)
    os.system(f"./{NCPacExeName}")
    # Move .xyz, FEATURES.csv to {finalDir}
    if not exists(f"{confDir}/od_FEATURESET.csv"):  # If execution unsuccessful due to being a BNP instead of TNP
        open(f"{finalDataPath}/Features/{confID}.csv", 'w').close()
    else:
        shutil.copy(f"{confDir}/{confID}.xyz", f"{finalDataPath}/Structures")
        shutil.copy(f"{confDir}/od_FEATURESET.csv", f"{finalDataPath}/Features/{confID}.csv")
    # Remove unnecessary files
    for f in glob(f"*.mod"): os.remove(f)
    for f in glob(f"fort.*"): os.remove(f)
    for f in glob(f"ov_*"): os.remove(f)
    for f in glob(f"od_*"): 
        if f != 'od_FEATURESET.csv': os.remove(f)
    open(f"{confDir}/{doneFile}", 'w').close()
    print(f"   {confID} Done!")


def runNCPacParallel(remainingWork):
    if not isdir(finalDataPath): os.mkdir(finalDataPath)
    if not isdir(f"{finalDataPath}/Structures"): os.mkdir(f"{finalDataPath}/Structures")
    if not isdir(f"{finalDataPath}/Features"): os.mkdir(f"{finalDataPath}/Features")
    # Check if confDir still exists (could have been run already)
    with Pool() as p: p.map(runNCPac, remainingWork)


if __name__ == '__main__':
    if runTask == 'setupNCPac':
        workingList = setupNCPac()
        with open('workingList.pickle', 'wb') as f: pickle.dump(workingList, f)
    elif runTask == 'runNCPac':
        with open('workingList.pickle', 'rb') as f: workingList = pickle.load(f)

        remainingWork = []
        for workParam in workingList:
            if not exists(f"{workParam[0]}/{doneFile}"): remainingWork.append(workParam)

        if runParallel: 
            runNCPacParallel(remainingWork)
        else:
            for work in remainingWork:
                runNCPac(work, verbose=True)
    print("All DONE!")
