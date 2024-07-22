# Goal: Compute the 3D box-counting dimension of nanoparticles, or any other objects consisting of sphere-like components
# (atoms, coarse-grained atoms, etc)

from os import listdir
from os.path import isdir
from natsort import natsorted
import numpy as np
from sphractal.src.sphractal import *


testCaseName, runPointCloudBoxCnt, runExactSurfBoxCnt, procUnit = 'PdNPs', True, False, 'cpu'
boxLenConc, atomConc, boxConc = False, True, False

boxLenRange = 'Trimmed'
vis, writeBox, rmInSurf, verbose, calcAvgDuration, findSurfOption = True, True, True, True, False, 'alphaShape'

GRID_NUM = 1024  # 4056 sufficient for < 3500 points for disPd1, Max 1048576 for laptop before RAM runs out
NUM_SPHERE_POINT = 300  # Number of points to be fitted onto each atomic sphere
ALPHA_MULT = 2.5

PROJECT_DIR = '/scratch/vp91/jt5911'  # To be modified when run different machines
BIN_IM_BC_EXE_DIR = f"{PROJECT_DIR}/sphractal/bin"
OUTPUT_DIR = f"{PROJECT_DIR}/outputs"


@utils.estDuration
def calcBLCN(atoms, eleSet, atomPosMinMax):
    """For measuring of time complexity, turn on calcBL for calcBoxCntDim for actual runs on NP simulations."""
    minAtomRad, maxAtomRad, minRadEle, maxRadEle = 10.0, 0.0, '', ''
    for atomEle in eleSet:
        if constants.ATOMIC_RAD_DICT[atomEle] < minAtomRad:
            minAtomRad, minRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
        if constants.ATOMIC_RAD_DICT[atomEle] > maxAtomRad:
            maxAtomRad, maxRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
    utils.findNN(atoms, atomPosMinMax, maxAtomRad, calcBL=True)
    avgBondLens = np.array([atom.avgBondLen for atom in atoms])
    coordNums = np.array([len(atom.neighs) for atom in atoms])
    return avgBondLens, coordNums


@utils.estDuration
def calcBoxCntDim(atoms, maxDimDiff, eleSet, atomPosMinMax, testCase,
                  findSurfOption='alphaShape', calcBL=False):
    """For running on NCI HPC facilities (Gadi) where data are stored."""
    minAtomRad, maxAtomRad, minRadEle, maxRadEle = 10.0, 0.0, '', ''
    for atomEle in eleSet:
        if constants.ATOMIC_RAD_DICT[atomEle] < minAtomRad:
            minAtomRad, minRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
        if constants.ATOMIC_RAD_DICT[atomEle] > maxAtomRad:
            maxAtomRad, maxRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
    utils.findNN(atoms, atomPosMinMax, maxAtomRad, calcBL=calcBL)
    utils.findSurf(atoms, option=findSurfOption, alpha=2*constants.ATOMIC_RAD_DICT[minRadEle])
    minBoxLen, maxBoxLen = minAtomRad / 4, minAtomRad
    # TODO: Remove @estDuration first!
    scaleChange, countChange = boxCnt.getSphereBoxCnts(atomList, maxDimDiff, (minBoxLen, maxBoxLen),
                                                       atomPosMinMax[:3], OUTPUT_DIR, testCase,
                                                       rmInSurf=True, writeBox=True, verbose=False,
                                                       boxLenConc=False, atomConc=True,
                                                       boxConc=False)
    r2score, boxCntDim, slopeCI = boxCnt.findSlope(scaleChange, countChange,
                                                   OUTPUT_DIR, testCase, boxLenRange='Trimmed',
                                                   visReg=True, saveFig=True, showPlot=False)
    avgBondLens = np.array([atom.avgBondLen for atom in atoms]) if calcBL else None
    return r2score, boxCntDim, slopeCI, avgBondLens


if __name__ == '__main__':
    testCases = []
    for NPname in listdir(f"testCases/{testCaseName}"):
        if not isdir(f"testCases/{testCaseName}/{NPname}"): testCases.append(f"testCases/{testCaseName}/{NPname}")
        else: testCases.extend(natsorted([f"testCases/{NPname}/{i}" for i in listdir(f"testCases/{NPname}")]))

    # fracDimNPdict = {}
    boxCntDimPrev = 0.0
    for testCase in testCases:
        if 'OT' not in testCase: continue  # Debugging
        atomList, maxDimDiff, eleSet, minMaxXYZs = utils.readXYZ(testCase)
        testCase = testCase.split('/')[-1][:-4]
        if verbose: print(f"\nTest case: {testCase}")
        minAtomRad, maxAtomRad, minRadEle, maxRadEle = 10.0, 0.0, '', ''
        for atomEle in eleSet:
            if constants.ATOMIC_RAD_DICT[atomEle] < minAtomRad:
                minAtomRad, minRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
            if constants.ATOMIC_RAD_DICT[atomEle] > maxAtomRad:
                maxAtomRad, maxRadEle = constants.ATOMIC_RAD_DICT[atomEle], atomEle
        utils.findNN(atomList, minMaxXYZs, maxAtomRad)
        # ALPHA values that match the results from NCPac's cone algorithm (55 degrees as parameter):
        # > 2.0 (most), 2.2-5.1 (DHSX1T000), 2.1-2.8 (RDSXT000, CUS1T000) for perfect polyhedron NPs
        # 2.1-2.7, 2.4-2.6 (1 atom diff), 2.5 (1 atom diff) for disordered NPs
        # 2.38 (maximum 12 atoms diff) for heated NPs (RDSXTX23, THSXTX23)
        # Use metallic diameter of the smallest atom (Could potentially consider bond contraction in future)
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
        print(f"{testCase}\t\tD_Box: {boxCntDim:.4f} [{boxCntDimCI[0]:.4f}, {boxCntDimCI[1]:.4f}]\t\tR2: {r2score:.4f}"
              f"\t\tTime: {duration:.4f}\t\tDiff: {abs(boxCntDim - boxCntDimPrev):.4f}")
        boxCntDimPrev = boxCntDim
    #     break
    #     fracDimDict = {'boxCntDim': boxCntDim, 'lowCI': boxCntDimCI[0], 'upCI': boxCntDimCI[1], 'R2': r2score, 't': duration}
    #     fracDimNPdict[testCase] = fracDimDict

    # import pandas as pd
    # fracDimDF = pd.DataFrame.from_dict(fracDimNPdict, orient='index')
    # fracDimDF['NPname'] = fracDimDF.index
    # fracDimDF['NPshape'] = fracDimDF['NPname'].apply(lambda x: x[:2])
    # fracDimDF['NPtemp'] = fracDimDF['NPname'].apply(lambda x: x[-3:])
    # # TH = 4 {111}
    # # CU = 6 {100}
    # # OT = 8 {111}
    # # RD = 12 {110}
    # # TO = 8 {111}, 6 {100}
    # # CO = 8 {111}, 6 {100}
    # # DH = 10 {111}, 5 {100}  # Ino
    # # IC = 20 {111}
    # numFaceOrderRank = {'TH': 1, 'CU': 2, 'OT': 3, 'RD': 4, 'TO': 5, 'CO': 6, 'DH': 7, 'IC': 8, 'di': 9}
    # fracDimDF['orderRank'] = fracDimDF['NPshape'].map(numFaceOrderRank)
    # fracDimDF.sort_values(by='orderRank')
    # fracDimIdealNPDF = fracDimDF[fracDimDF['NPtemp'] == '000']
    # fracDimIdealNPDF.groupby(by='NPshape').mean().sort_values(by='orderRank')
    # fracDimHeatedNPDF = fracDimDF[fracDimDF['NPtemp'] == '323']
    # fracDimHeatedNPDF.groupby(by='NPshape').mean().sort_values(by='orderRank')
    #
    # # import pickle
    # # with open(f"{boxLenRange}.pickle", 'wb') as f: pickle.dump(fracDimDF, f)
    # # with open(f"{boxLenRange}.pickle", 'rb') as f: fracDimDF = pickle.load(f)
    # # print(fracDimDF)
    #
    # if calcAvgDuration:
    #     avgDuration, repeatNum = 0, 10
    #     for i in range(repeatNum):
    #         _, duration = surfBoxCnt(atomList, maxDimDiff, (minBoxLen, maxBoxLen), minMaxXYZs[:3], OUTPUT_DIR, testCase, writeBox=True, verbose=False)
    #         avgDuration += duration
    #     print(f"Average duration: {avgDuration / repeatNum:.6f} s")
    #
    # if verbose:
    #     print(f"Change in\n\tScale: {scaleChange}\n\tCounts: {countChange}")
    #     print(f"Coefficient of determination (R2): {r2score:.3f}\
    #           \nEstimated box-counting dimension (D_Box): {boxCntDim:.3f}\
    #           \n{CONF_INT_PERC}% Confidence interval: [{boxCntDimCI[0]:.3f}, {boxCntDimCI[1]:.3f}]")
