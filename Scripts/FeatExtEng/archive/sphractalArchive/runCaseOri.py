# Goal: Compute the 3D box-counting dimension of nanoparticles, or any other objects consisting of sphere-like components
# (atoms, coarse-grained atoms, etc)

from os import listdir
from os.path import isdir
from natsort import natsorted
import numpy as np
from sphractal.src.sphractal import *
from nvtx import annotate


testCaseName, runPointCloudBoxCnt, runExactSurfBoxCnt, procUnit = 'PtAuNPs', False, True, 'cpu'
boxLenConc, atomConc, boxConc = False, False, False

boxLenRange = 'Trimmed'
vis, writeBox, rmInSurf, verbose, calcAvgDuration, findSurfOption = True, True, True, True, False, 'alphaShape'

GRID_NUM = 1024
NUM_SPHERE_POINT = 300  # Number of points to be fitted onto each atomic sphere
ALPHA_MULT = 2.5

PROJECT_DIR = '/scratch/vp91/jt5911'  # To be modified when run different machines
BIN_IM_BC_EXE_DIR = f"{PROJECT_DIR}/sphractal/bin"
OUTPUT_DIR = f"{PROJECT_DIR}/testOutputs"


@annotate('runCase',color='red')
@utils.estDuration
def runCase(testCase):
    atomList, maxDimDiff, eleSet, minMaxXYZs = utils.readXYZ(testCase)
    testCase = testCase.split('/')[-1][:-4]
    print(f"\n{testCase}")
    minAtomRad, maxAtomRad, minRadEle, maxRadEle = 10.0, 0.0, '', ''
    for atomEle in eleSet:
        if constants.ATOMIC_RAD_DICT[atomEle] < minAtomRad:
            minAtomRad, minRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
        if constants.ATOMIC_RAD_DICT[atomEle] > maxAtomRad:
            maxAtomRad, maxRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
    utils.findNN(atomList, minMaxXYZs, maxAtomRad)
    utils.findSurf(atomList, option=findSurfOption, alpha=ALPHA_MULT*utils.ATOMIC_RAD_DICT[minRadEle])

    if runPointCloudBoxCnt:
        (scaleChange, countChange), duration = boxCnt.getVoxelBoxCnts(atomList, 
                                                                      npName=testCase, writeFileDir=OUTPUT_DIR,
                                                                      exeDir=BIN_IM_BC_EXE_DIR,
                                                                      numPoint=NUM_SPHERE_POINT, gridNum=GRID_NUM, 
                                                                      procUnit=procUnit, rmInSurf=rmInSurf, 
                                                                      vis=vis, verbose=verbose, 
                                                                      genPCD=False)
        r2score, boxCntDim, boxCntDimCI = boxCnt.findSlope(scaleChange[::-1], countChange[::-1],
                                                           OUTPUT_DIR, testCase, boxLenRange,
                                                           visReg=vis, saveFig=vis, showPlot=False)

    if runExactSurfBoxCnt:
        if boxLenRange == 'FullRange': 
            minBoxLen, maxBoxLen = minAtomRad / 4, minAtomRad * 2
        else: 
            minBoxLen, maxBoxLen = minAtomRad / 4, minAtomRad
        (scaleChange, countChange), duration = boxCnt.getSphereBoxCnts(atomList, maxDimDiff, (minBoxLen, maxBoxLen),
                                                                       minMaxXYZs[:3], OUTPUT_DIR, testCase,
                                                                       rmInSurf=rmInSurf, writeBox=writeBox, verbose=verbose,
                                                                       boxLenConc=boxLenConc, atomConc=atomConc, boxConc=boxConc)
        r2score, boxCntDim, boxCntDimCI = boxCnt.findSlope(scaleChange, countChange, 
                                                           OUTPUT_DIR, testCase, boxLenRange, 
                                                           visReg=vis, saveFig=vis, showPlot=False)
    print(f"    D_Box: {boxCntDim:.4f} [{boxCntDimCI[0]:.4f}, {boxCntDimCI[1]:.4f}]    R2: {r2score:.4f}")
    return


if __name__ == '__main__':
    testCases = []
    for NPname in listdir(f"testCases/{testCaseName}"):
        if not isdir(f"testCases/{testCaseName}/{NPname}"): testCases.append(f"testCases/{testCaseName}/{NPname}")
        else: testCases.extend(natsorted([f"testCases/{NPname}/{i}" for i in listdir(f"testCases/{NPname}")]))

    for testCase in testCases:
        # if 'OT' not in testCase: continue  # Debugging
        _, duration = runCase(testCase) 
        print(f"Duration: \t{duration:.4f} s")
        break
