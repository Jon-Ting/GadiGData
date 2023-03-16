import os
import shutil

ELE1, ELE2 = 'Co', 'Pd'
cutOffDict = {'AuAu': 3.5, 'AuPt': 3.5, 'AuPd': 3.5, 'AuCo': 3.4,
              'PtAu': 3.5, 'PtPt': 3.5, 'PtPd': 3.5, 'PtCo': 3.4,
              'PdAu': 3.5, 'PdPt': 3.5, 'PdPd': 3.5, 'PdCo': 3.4,
              'CoAu': 3.4, 'CoPt': 3.4, 'CoPd': 3.4, 'CoCo': 3.3}
PROJECT, USER_NAME = 'q27', 'jt5911'
targetDir = f"/scratch/{PROJECT}/{USER_NAME}/{ELE1}{ELE2}"
NCPacExeName, NCPacInpName = 'NCPac.exe', 'NCPac.inp'
path2NCPacExe = f"/g/data/{PROJECT}/{USER_NAME}/Scripts/FeatExtEng/NCPac/{NCPacExeName}"
path2NCPacInp = f"/g/data/{PROJECT}/{USER_NAME}/Scripts/FeatExtEng/NCPac/{NCPacInpName}"
headerLine = f"CSIRO Nanostructure Databank - {ELE1}{ELE2} Nanoparticle Data Set"

for confID in os.listdir(targetDir):
    confDir = f"{targetDir}/{confID}"
    shutil.copy(path2NCPacExe, f"{confDir}/{NCPacExeName}")
    with open(path2NCPacInp, 'r') as f1: linesNCPac = f1.readlines()
    with open(f"{confDir}/{NCPacInpName}", 'w') as f2:
        linesNCPac[0] =  f"{confID}.xyz        - name of xyz input file                                              [in_filexyz]\n"
        linesNCPac[5] =  f"{ELE1} {cutOffDict[ELE1*2]} {cutOffDict[ELE1+ELE2]}        - NN unique cutoff matrix (line1 type1,r1r1,r1r2, line2 type2 r2r2)   [in_cutoff(i,j)]\n"
        linesNCPac[6] =  f"{ELE2} {cutOffDict[ELE2*2]}\n"
        f2.writelines(linesNCPac)

