from glob import glob
from multiprocessing import Pool
import os
from os.path import isdir, exists
import pandas as pd
import re
import shutil
import tarfile
from zipfile import ZipFile
from filtRedund import runFilter


ELE1, ELE2 = 'Co', 'Pd'
runTask = 'runNCPac'  # 'setupNCPac'or 'filtRedund' or 'runNCPac'
startNPidx = 136996  # Only used for 'filtRedund'
NPsize = 'small'
replace, runParallel, verbose = False, True, True
sourceDirs = ['L10', 'L12', 'RAL','RCS', 'CS']

cutOffDict = {'AuAu': 3.5, 'AuPt': 3.5, 'AuPd': 3.5, 'AuCo': 3.4,
              'PtAu': 3.5, 'PtPt': 3.5, 'PtPd': 3.5, 'PtCo': 3.4,
              'PdAu': 3.5, 'PdPt': 3.5, 'PdPd': 3.5, 'PdCo': 3.4,
              'CoAu': 3.4, 'CoPt': 3.4, 'CoPd': 3.4, 'CoCo': 3.3}
PROJECT, USER_NAME = 'q27', 'jt5911'
targetDir = f"/scratch/{PROJECT}/{USER_NAME}/{ELE1}{ELE2}"
numFramePerNP = 101
totalNPnum = 1668
zFillNum = 6
doneFile = 'DONE.txt'
outMDfile = 'MDout.csv'
NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
path2NCPacExe = f"/g/data/{PROJECT}/{USER_NAME}/Scripts/FeatExtEng/NCPac/{NCPacExeName}"
path2NCPacInp = f"/g/data/{PROJECT}/{USER_NAME}/Scripts/FeatExtEng/NCPac/{NCPacInpName}"
headerLine = f"CSIRO Nanostructure Databank - {ELE1}{ELE2} Nanoparticle Data Set"
if NPsize == 'small': 
    sourcePaths = [f"/g/data/{PROJECT}/{USER_NAME}/PostSim/{sourceDir}" for sourceDir in sourceDirs]
    npCnt = 0
else: 
    sourcePaths = [f"/scratch/{PROJECT}/{USER_NAME}/BNP_MDsim/{sourceDir}50+" for sourceDir in sourceDirs]
    npCnt = 1668  # Last NP should be 4788
    

def setupNCPac(npCnt=0, replace=False, verbose=False):
    if verbose: print(f"Copying {ELE1}{ELE2} xyz files to individual directories and relabelling numerically...")
    if not isdir(targetDir): os.mkdir(targetDir)
    if not exists(f"{targetDir}/{outMDfile}"): 
        with open(f"{targetDir}/{outMDfile}", 'w') as f: f.write('confID,T,P,PE,KE,TE\n')

    for sourcePath in sourcePaths:
        if verbose: print(f"  Source path: {sourcePath}") 
        for eleSubDir in os.listdir(sourcePath):
            if ELE1 not in eleSubDir: continue
            if verbose: print(f"    Element subdirectory: {eleSubDir}") 
            for NPdir in os.listdir(f"{sourcePath}/{eleSubDir}"):
                if ELE2 not in NPdir: continue
                print(f"      Nanoparticle: {NPdir}")
                NPdirPath = f"{sourcePath}/{eleSubDir}/{NPdir}"
            
                if not exists(f"{NPdirPath}/{doneFile}") or replace:
                    if exists(f"{NPdirPath}/{NPdir}S2.zip"):
                        with ZipFile(f"{NPdirPath}/{NPdir}S2.zip", 'r') as f: f.extractall(f"{NPdirPath}/")
                    elif exists(f"{NPdirPath}/{NPdir}S2.tar.gz"):
                        with tarfile.open(f"{NPdirPath}/{NPdir}S2.tar.gz", 'r') as f: f.extractall(f"{NPdirPath}/")
                    else: raise Exception(f"        {NPdir} doesn't have S2 files!")
                    if verbose: print('        Extracted Stage 2 files...')
                else:
                    npCnt += 1
                    continue
                 
                allS2NPs = [bnp for bnp in os.listdir(f"{NPdirPath}/{NPdir}S2") if 'min' in bnp]
                confCnt = npCnt*numFramePerNP
                for NPconf in sorted(allS2NPs, key=lambda key: [int(i) for i in re.findall('min.([0-9]+)', key)]):
                    oriFilePath = f"{NPdirPath}/{NPdir}S2/{NPconf}"
                    confID = str(confCnt).zfill(zFillNum)
                    if verbose: print(f"          Conformation ID: {confID}")
                    confDir = f"{targetDir}/{confID}"
                    if not isdir(f"{targetDir}/{confID}"): os.mkdir(confDir)

                    if verbose: print('            Replacing header...')
                    with open(oriFilePath, 'r') as f1:
                        with open(f"{confDir}/{confID}.xyz", 'w') as f2:
                            f2.write(f1.readline())
                            f1.readline()  # Replace second line with CSIRO header
                            f2.write(f"{headerLine} - {NPdir}\n")
                            f2.write(''.join([line for line in f1.readlines()]))

                    if verbose: print('            Copying files...')
                    shutil.copy(path2NCPacExe, f"{confDir}/{NCPacExeName}")
                    with open(path2NCPacInp, 'r') as f1: linesNCPac = f1.readlines()
                    with open(f"{confDir}/{NCPacInpName}", 'w') as f2:
                        linesNCPac[0] =  f"{confID}.xyz        - name of xyz input file                                              [in_filexyz]\n"
                        linesNCPac[5] =  f"{ELE1} {cutOffDict[ELE1*2]} {cutOffDict[ELE1+ELE2]}        - NN unique cutoff matrix (line1 type1,r1r1,r1r2, line2 type2 r2r2)   [in_cutoff(i,j)]\n"
                        linesNCPac[6] =  f"{ELE2} {cutOffDict[ELE2*2]}\n"
                        f2.writelines(linesNCPac)
                    confCnt += 1
                
                if verbose: print('        Extracting MD output...')
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
            
                if verbose: print('        Cleaning up directory...')
                shutil.rmtree(f"{NPdirPath}/{NPdir}S2")
                open(f"{NPdirPath}/{doneFile}", 'w').close()
                if verbose: print(f"        {NPdir} Done!")


def runFiltRedund(startNPidx=0, verbose=False):
    NPfiltNames = runFilter(targetDir=targetDir, startNPidx=startNPidx, verbose=verbose)
    if not os.path.isdir(f"{targetDir}/REDUNDANT"): 
        os.mkdir(f"{targetDir}/REDUNDANT")
        for npName in NPfiltNames: shutil.move(f"{targetDir}/{npName}", f"{targetDir}/REDUNDANT")
    shutil.rmtree(f"{targetDir}/REDUNDANT")


def runNCPac(work, verbose=False):
    confDir, confID = work
    # Execute NCPac.exe for every directories
    print(f"  Nanoparticle: {confID}...")
    os.chdir(confDir)
    os.system(f"./{NCPacExeName}")
    if not exists(f"{confDir}/od_FEATURESET.csv"): 
        print(f"    {confID} is a MNP!")
        open(f"{confDir}/od_FEATURESET.csv", 'w').close()  # If NCPac execution somehow fails (MNP instead of BNP)
    # Remove unnecessary files
    for f in glob(f"*.mod"): os.remove(f)
    for f in glob(f"fort.*"): os.remove(f)
    for f in glob(f"ov_*"): os.remove(f)
    for f in glob(f"od_*"): 
        if f != 'od_FEATURESET.csv': os.remove(f)
    open(f"{confDir}/{doneFile}", 'w').close()
    if verbose: print(f"    {confID} Done!")


def runNCPacParallel(remainingWork, verbose=False):
    with Pool() as p: p.map(runNCPac, remainingWork)


if __name__ == '__main__':
    if runTask == 'setupNCPac': # Copy necessary files to run NCPac to destination
        setupNCPac(npCnt, replace=replace, verbose=verbose)
    elif runTask == 'filtRedund': # Remove redundant frames from DAP data
        runFiltRedund(startNPidx=startNPidx, verbose=verbose) 
    elif runTask == 'runNCPac': # Run NCPac for each BNP
        workingList = [(f"{targetDir}/{str(i).zfill(zFillNum)}", str(i).zfill(zFillNum)) for i in range(totalNPnum*numFramePerNP)]
        if verbose: print(f"Running NCPac for {ELE1}{ELE2} nanoparticles...")
        if replace: workingList = [workParam for workParam in workingList if exists(f"{workParam[0]}")]
        else: workingList = [workParam for workParam in workingList if exists(f"{workParam[0]}") and not exists(f"{workParam[0]}/{doneFile}")]
        if runParallel:
            with Pool() as p: p.map(runNCPac, workingList)
        else:
            for work in workingList: runNCPac(work, verbose=verbose)
    print("All DONE!")

