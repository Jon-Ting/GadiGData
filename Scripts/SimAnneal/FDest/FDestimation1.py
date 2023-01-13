#!/home/jonting/anaconda3/bin/python
# Goal: Compute the box-counting dimension of nanoparticles, or any other objects consisting of sphere-like components
# (atoms, coarse-grained atoms, etc)
# Author: Jonathan Yik Chang Ting
# Date: 7/11/2022
# TODO:
# - find justification for alpha value, or look into potentially better findSurf() algorithm
# - Try parallelising over atom and then merging (but not much speed gain expected due to limited computing involved)

from collections import defaultdict
from itertools import product
from math import ceil, floor, log10, sqrt
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import ConvexHull, Delaunay
from statsmodels.api import OLS, add_constant
from time import time


MAX_RES = 6
BOX_LEN_INTERVAL = 20

BOX_LEN_THRESH = 5 * BOX_LEN_INTERVAL  # Should be when scanBoxLen*sqrt(3) is approximately equal to atomic radius
R2_THRESHOLD = 0.99
CONF_INT_PERC = 95  # %
ALPHA_CONF_INT = 1 - CONF_INT_PERC/100

# ALPHA values that match the results from NCPac's cone algorithm (55 degrees as parameter):
# > 2.0 (most), 2.2-5.1 (DHSX1T000), 2.1-2.8 (RDSXT000, CUS1T000) for perfect polyhedron NPs
# 2.1-2.7, 2.4-2.6 (1 atom diff), 2.5 (1 atom diff) for disordered NPs
# 2.38 (maximum 12 atoms diff) for heated NPs (RDSXTX23, THSXTX23)
ALPHA = 2.38
MIN_VAL_FROM_BOUND = 5.0  # Angstrom
NN_DIST_THRESH = 4.0  # Angstrom
BULK_CN = 12
NUM_SCAN_THRESH_MP = 10000  # Number of scans to switch to concurrent code
# Computed from theoretical models - E. Clementi; D.L. Raimondi; W.P. Reinhardt (1967) "Atomic Screening Constants from
# SCF Functions. II. Atoms with 37 to 86 Electrons." The Journal of Chemical Physics. 47 (4): 1300-1307
ATOMIC_RAD_DICT = {'Co': 1.52, 'Pd': 1.69, 'Pt': 1.77, 'Au': 1.74}  # Types denoted by integer in .LMP file
ATOMIC_RAD_DICT = {'Co': 1.672, 'Pd': 1.859, 'Pt': 1.947, 'Au': 1.914}  # 10% larger
# N.N. Greenwood; A. Earnshaw (1997) "Chemistry of Elements (2nd ed.)" Butterworth-Heinemann
# METALLIC_RAD_DICT = {'Co': 1.25, 'Pd': 1.37, 'Pt': 1.385, 'Au': 1.44}


class Atom(object):
    def __init__(self, atomIdx, eleXYZ):
        self.ID = atomIdx
        self.ele = eleXYZ[-4]
        self.X, self.Y, self.Z = float(eleXYZ[-3]), float(eleXYZ[-2]), float(eleXYZ[-1])
        self.neighs = []
        self.isSurf = 0  # int(eleXYZ[4]) from 'ov_SURF_layer.xyz'
        # self.TypeID = None
        # self.mass = 1


def estDuration(func):
    def wrap(*arg, **kwargs):
        start = time()
        result = func(*arg, **kwargs)
        end = time()
        duration = end - start
        # print(f"*** Function {func.__name__} duration: {duration:.6f} s\n")
        return result, duration
    return wrap


def readXYZ(filePath):
    atomList = []
    maxX, maxY, maxZ, minX, minY, minZ = 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0
    numLinesSkip = 9 if '.lmp' in filePath else 2
    with open(filePath, 'r') as f:
        for (i, line) in enumerate(f):
            if i < numLinesSkip: continue
            atom = Atom(i -numLinesSkip, line.split())
            if atom.X > maxX: maxX = atom.X
            if atom.X < minX: minX = atom.X
            if atom.Y > maxY: maxY = atom.Y
            if atom.Y < minY: minY = atom.Y
            if atom.Z > maxZ: maxZ = atom.Z
            if atom.Z < minZ: minZ = atom.Z
            atomList.append(atom)
    maxDimDiff = max(maxX - minX, maxY - minY, maxZ - minZ)
    return atomList, maxDimDiff, (minX, minY, minZ, maxX, maxY, maxZ)


def findNN(atomList, minX, minY, minZ, maxX, maxY, maxZ):

    # Simpler but slower approach
    # for i, atom1 in enumerate(atomList):
    #     for j in range(i + 1, len(atomList)):
    #         atom2 = atomList[j]
    #         diffX, diffY, diffZ = abs(atom1.X - atom2.X), abs(atom1.Y - atom2.Y), abs(atom1.Z - atom2.Z)
    #         if diffX < NN_DIST_THRESH and diffY < NN_DIST_THRESH and diffZ < NN_DIST_THRESH:
    #             if diffX * diffX + diffY * diffY + diffZ * diffZ < (ATOMIC_RAD_DICT[atom1.ele]*2) ** 2:
    #                 atom1.neighs.append(j)
    #                 atom2.neighs.append(i)

    allDirections = [dirVec for dirVec in product(range(-1, 2), repeat=3)]
    stepSize = ceil(NN_DIST_THRESH)
    numX, numY, numZ = ceil((maxX-minX) / stepSize), ceil((maxY-minY) / stepSize), ceil((maxZ-minZ) / stepSize)
    boxes = [[[[] for z in range(numZ)] for y in range(numY)] for x in range(numX)]
    for (i, atom1) in enumerate(atomList):
        x, y, z = floor((atom1.X-minX) / stepSize), floor((atom1.Y-minY) / stepSize), floor((atom1.Z-minZ) / stepSize)
        for (dirX, dirY, dirZ) in allDirections:
            if 0 <= x + dirX < numX and 0 <= y + dirY < numY and 0 <= z + dirZ < numZ:
                for j in boxes[x + dirX][y + dirY][z + dirZ]:
                    atom2 = atomList[j]
                    diffX, diffY, diffZ = abs(atom1.X - atom2.X), abs(atom1.Y - atom2.Y), abs(atom1.Z - atom2.Z)
                    if diffX * diffX + diffY * diffY + diffZ * diffZ < (ATOMIC_RAD_DICT[atom1.ele]*2) ** 2:
                        atom1.neighs.append(j)
                        atom2.neighs.append(i)
        boxes[x][y][z].append(i)


def findSurf(atomList, option='alphaShape', alpha=2.38):
    """
    convexHull:
        - Tend to identify less than actual surface atoms
    numNeigh:
        - Tend to identify more than actual surface atoms
    alphaShape:
        - Generalisation of convex hull
        - Algorithm modified from https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d
        - Radius of the sphere fitting inside the tetrahedral < alpha (http://mathworld.wolfram.com/Circumsphere.html)

    Alternatives:
    - https://dl.acm.org/doi/abs/10.1145/2073304.2073339
    - https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.25384
    - https://www.jstage.jst.go.jp/article/tmrsj/45/4/45_115/_article
    """
    if option == 'convexHull':
        atoms = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList])
        for atomIdx in ConvexHull(atoms).vertices: atomList[atomIdx].isSurf = 1
    elif option == 'numNeigh':
        for atom in atomList:
            if len(atom.neighs) < BULK_CN: atom.isSurf = 1
    elif option == 'alphaShape':
        atoms = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList])
        tetraVtxsIdxs = Delaunay(atoms).simplices

        # Find radius of the circumsphere
        tetraVtxs = np.take(atoms, tetraVtxsIdxs, axis=0)
        norm2 = np.sum(tetraVtxs ** 2, axis=2)[:, :, None]
        ones = np.ones((tetraVtxs.shape[0], tetraVtxs.shape[1], 1))
        a = np.linalg.det(np.concatenate((tetraVtxs, ones), axis=2))
        Dx = np.linalg.det(np.concatenate((norm2, tetraVtxs[:, :, [1, 2]], ones), axis=2))
        Dy = np.linalg.det(np.concatenate((norm2, tetraVtxs[:, :, [0, 2]], ones), axis=2))
        Dz = np.linalg.det(np.concatenate((norm2, tetraVtxs[:, :, [0, 1]], ones), axis=2))
        c = np.linalg.det(np.concatenate((norm2, tetraVtxs), axis=2))
        with np.errstate(divide='ignore', invalid='ignore'): r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4*a*c) / (2*np.abs(a))

        # Find tetrahedrons and triangles
        tetras = tetraVtxsIdxs[r < alpha, :]
        triComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        tris = tetras[:, triComb].reshape(-1, 3)
        tris = np.sort(tris, axis=1)

        # Remove triangles that occurs twice, because they are internal
        trisDict = defaultdict(int)
        for tri in tris: trisDict[tuple(tri)] += 1
        tris = [tri for tri in trisDict if trisDict[tri] == 1]
        npVtxs = np.unique(np.concatenate(np.array(tris)))
        for atomIdx in npVtxs: atomList[atomIdx].isSurf = 1


def getNearFarCoord(scanBoxIdx, scanBoxLen, lowBound, atomCoord):
    scanBoxMax = lowBound - MIN_VAL_FROM_BOUND + (scanBoxIdx+1)*scanBoxLen
    scanBoxMin = scanBoxMax - scanBoxLen
    if atomCoord < scanBoxMin: scanBoxNear, scanBoxFar = scanBoxMin, scanBoxMax
    elif atomCoord > scanBoxMax: scanBoxNear, scanBoxFar = scanBoxMax, scanBoxMin
    else:
        scanBoxNear = atomCoord
        scanBoxFar = scanBoxMin if scanBoxMax - atomCoord < scanBoxLen / 2 else scanBoxMax  # Always a corner
    return scanBoxNear, scanBoxFar


# def scanBoxes(atomBoxIdxs, scanBoxDir, magnFac, scanBoxLen, minVals, atom, bulkSet, surfSet, overlapSet):
#     # Parallel running of box scanning around a given atom
#     scanBoxIdxX, scanBoxIdxY, scanBoxIdxZ = atomBoxIdxs[0] + scanBoxDir[0], atomBoxIdxs[1] + scanBoxDir[1], \
#                                             atomBoxIdxs[2] + scanBoxDir[2]
#     scanBoxIdxs = (scanBoxIdxX, scanBoxIdxY, scanBoxIdxZ)
#     if any(i < 0 or i >= magnFac for i in scanBoxIdxs): return
#
#     scanBoxNearX, scanBoxFarX = getNearFarCoord(scanBoxIdxX, scanBoxLen, minVals[0], atom.X)
#     scanBoxNearY, scanBoxFarY = getNearFarCoord(scanBoxIdxY, scanBoxLen, minVals[1], atom.Y)
#     scanBoxNearZ, scanBoxFarZ = getNearFarCoord(scanBoxIdxZ, scanBoxLen, minVals[2], atom.Z)
#     distNear = sqrt((atom.X-scanBoxNearX)**2 + (atom.Y-scanBoxNearX)**2 + (atom.Z-scanBoxNearZ)**2)
#     distFar = sqrt((atom.X-scanBoxFarX)**2 + (atom.Y-scanBoxFarY)**2 + (atom.Z-scanBoxFarZ)**2)
#
#     overlapAddSet, surfAddSet, bulkAddSet, cntBulk, cntSurf, cntOverlap = set(), set(), set(), 0, 0, 0
#     atomRad = ATOMIC_RAD_DICT[atom.ele]
#     if not atom.isSurf:
#         if distFar < atomRad:
#             if scanBoxIdxs in overlapSet or scanBoxIdxs in bulkSet: return
#             elif scanBoxIdxs in surfSet:
#                 overlapAddSet.add(scanBoxIdxs)
#                 cntOverlap += 1
#             else:
#                 bulkAddSet.add(scanBoxIdxs)
#                 cntBulk += 1
#     else:
#         if distNear < atomRad < distFar:
#             if scanBoxIdxs in overlapSet or scanBoxIdxs in surfSet: return
#             elif scanBoxIdxs in bulkSet:
#                 overlapAddSet.add(scanBoxIdxs)
#                 cntBulk -= 1
#                 cntOverlap += 1
#                 cntSurf += 1
#             else:
#                 surfAddSet.add(scanBoxIdxs)
#                 cntSurf += 1
#         elif distFar < atomRad:
#             if scanBoxIdxs in overlapSet or scanBoxIdxs in bulkSet: return
#             elif scanBoxIdxs in surfSet:
#                 overlapAddSet.add(scanBoxIdxs)
#                 cntOverlap += 1
#             else:
#                 bulkAddSet.add(scanBoxIdxs)
#                 cntBulk += 1
#     return overlapAddSet, surfAddSet, bulkAddSet, cntOverlap, cntSurf, cntBulk


def scanBoxes(boxRes, boxLen, atomList, minX, minY, minZ, verbose=False):
    magnFac = int(2 ** (boxRes / BOX_LEN_INTERVAL))
    scale = log10(magnFac / boxLen)
    scanBoxLen = boxLen / magnFac
    numScan = int(NN_DIST_THRESH / scanBoxLen)
    surfSet, bulkSet, cntSurf, cntBulk = set(), set(), 0, 0
    for atom in atomList:
        # if scanBoxLen > ATOMIC_RAD_DICT[atom.ele]:
        #     print('skip atom:', atom, scanBoxLen)
        #     continue
        if sum([atomList[neighIdx].isSurf for neighIdx in atom.neighs]) == 0: continue
        atomBoxIdxX = int((atom.X - minX + MIN_VAL_FROM_BOUND) / scanBoxLen)
        atomBoxIdxY = int((atom.Y - minY + MIN_VAL_FROM_BOUND) / scanBoxLen)
        atomBoxIdxZ = int((atom.Z - minZ + MIN_VAL_FROM_BOUND) / scanBoxLen)
        # if numScan > NUM_SCAN_THRESH_MP:
        #     # print('Parallelising...')
        #     scanBoxesInp = [((atomBoxIdxX, atomBoxIdxY, atomBoxIdxZ), scanBoxDir, magnFac, scanBoxLen,
        #                      (minX, minY, minZ), atom, overlapSet, surfSet, bulkSet) for scanBoxDir in
        #                     product(range(-numScan, numScan + 1), repeat=3)]
        #     with Pool() as pool:
        #         for scanResult in pool.starmap(scanBoxes, scanBoxesInp):  # starmap (not map) for >1 arguments
        #             if scanResult:
        #                 overlapSet.update(scanResult[0])
        #                 scanResult[1].difference_update(scanResult[0])
        #                 surfSet.update(scanResult[1])
        #                 scanResult[2].difference_update(scanResult[0])
        #                 bulkSet.update(scanResult[2])
        #                 cntOverlap += scanResult[3]
        #                 cntSurf += scanResult[4]
        #                 cntBulk += scanResult[5]
        #     continue

        # print('Running serial scans...')
        for k in range(-numScan, numScan + 1):
            scanBoxIdxX = atomBoxIdxX + k
            if scanBoxIdxX < 0 or scanBoxIdxX >= magnFac: continue
            scanBoxNearX, scanBoxFarX = getNearFarCoord(scanBoxIdxX, scanBoxLen, minX, atom.X)
            for l in range(-numScan, numScan + 1):
                scanBoxIdxY = atomBoxIdxY + l
                if scanBoxIdxY < 0 or scanBoxIdxY >= magnFac: continue
                scanBoxNearY, scanBoxFarY = getNearFarCoord(scanBoxIdxY, scanBoxLen, minY, atom.Y)
                for m in range(-numScan, numScan + 1):
                    scanBoxIdxZ = atomBoxIdxZ + m
                    if scanBoxIdxZ < 0 or scanBoxIdxZ >= magnFac: continue
                    scanBoxNearZ, scanBoxFarZ = getNearFarCoord(scanBoxIdxZ, scanBoxLen, minZ, atom.Z)
                    # Shorter but slower alternative
                    # scanBoxDirs = [scanBoxDir for scanBoxDir in product(range(-numScan, numScan + 1), repeat=3)]
                    # for scanBoxDir in scanBoxDirs:
                    #     scanBoxIdxX, scanBoxIdxY, scanBoxIdxZ = atomBoxIdxX + scanBoxDir[0], atomBoxIdxY + scanBoxDir[1], \
                    #                                             atomBoxIdxZ + scanBoxDir[2]
                    #     if any(i < 0 or i >= magnFac for i in (scanBoxIdxX, scanBoxIdxY, scanBoxIdxZ)): continue
                    #     scanBoxNearX, scanBoxFarX = getNearFarCoord(scanBoxIdxX, scanBoxLen, minX, atom.X)
                    #     scanBoxNearY, scanBoxFarY = getNearFarCoord(scanBoxIdxY, scanBoxLen, minY, atom.Y)
                    #     scanBoxNearZ, scanBoxFarZ = getNearFarCoord(scanBoxIdxZ, scanBoxLen, minZ, atom.Z)
                    scanBoxIdxs = (scanBoxIdxX, scanBoxIdxY, scanBoxIdxZ)
                    distNear = sqrt((atom.X-scanBoxNearX)**2 + (atom.Y-scanBoxNearY)**2 + (atom.Z-scanBoxNearZ)**2)
                    distFar = sqrt((atom.X-scanBoxFarX)**2 + (atom.Y-scanBoxFarY)**2 + (atom.Z-scanBoxFarZ)**2)
                    atomRad = ATOMIC_RAD_DICT[atom.ele]
                    if not atom.isSurf:
                        if distNear < atomRad < distFar or distFar < atomRad:
                            if scanBoxIdxs in bulkSet: continue
                            elif scanBoxIdxs in surfSet:
                                surfSet.remove(scanBoxIdxs)
                                bulkSet.add(scanBoxIdxs)
                                cntBulk += 1
                                cntSurf -= 1
                            else:
                                bulkSet.add(scanBoxIdxs)
                                cntBulk += 1
                    else:  # For surface atom
                        if distNear < atomRad < distFar:
                            if scanBoxIdxs in bulkSet or scanBoxIdxs in surfSet: continue
                            else:
                                surfSet.add(scanBoxIdxs)
                                cntSurf += 1
                        elif distFar < atomRad:
                            if scanBoxIdxs in bulkSet: continue
                            elif scanBoxIdxs in surfSet:
                                surfSet.remove(scanBoxIdxs)
                                bulkSet.add(scanBoxIdxs)
                                cntBulk += 1
                                cntSurf -= 1
                            else:
                                bulkSet.add(scanBoxIdxs)
                                cntBulk += 1
    if verbose: print(f"\tMagnification, Counts (bulk, surf):\t{magnFac} {cntBulk} {cntSurf}")
    return surfSet, bulkSet, cntSurf, cntBulk, scale


def surfBoxCnt(atomList, maxDimDiff, minX, minY, minZ, NPname='', writeBox=False, verbose=False):
    boxLen = maxDimDiff + MIN_VAL_FROM_BOUND * 2
    surfSets, bulkSets, cntSurfs, cntBulks, scaleChange = [], [], [], [], []
    scanBoxesInp = [(boxRes, boxLen, atomList, minX, minY, minZ, verbose) for boxRes in
                    range(1, MAX_RES*BOX_LEN_INTERVAL + 1) if boxRes >= BOX_LEN_THRESH]
    with Pool() as pool:
        for scanResult in pool.starmap(scanBoxes, scanBoxesInp):  # starmap (not map) for >1 arguments
            if not scanResult: print('Check scanResult!')
            surfSets.append(scanResult[0])
            bulkSets.append(scanResult[1])
            cntSurfs.append(scanResult[2])
            cntBulks.append(scanResult[3])
            scaleChange.append(scanResult[4])
    cntChange = [log10(cntS) if cntS != 0 else np.nan for cntS in cntSurfs]

    # Write out box counted for visualisation
    if writeBox:
        with open(f"ov_FRADIM_SURF_AREA_{NPname}.xyz", 'w') as f:
            for (i, boxRes) in enumerate(range(BOX_LEN_THRESH, MAX_RES*BOX_LEN_INTERVAL + 1)):
                scanBoxLen = boxLen / int(2 ** (boxRes / BOX_LEN_INTERVAL))

                if i != 0: f.write('\n')
                f.write(f"{len(atomList) + len(surfSets[i]) + len(bulkSets[i])}\n")
                for atom in atomList: f.write(f"\n{atom.ele}\t{atom.X} {atom.Y} {atom.Z}")
                for (boxIDX, boxIDY, boxIDZ) in surfSets[i]:
                    boxX = minX - MIN_VAL_FROM_BOUND + boxIDX*scanBoxLen + scanBoxLen/2
                    boxY = minY - MIN_VAL_FROM_BOUND + boxIDY*scanBoxLen + scanBoxLen/2
                    boxZ = minZ - MIN_VAL_FROM_BOUND + boxIDZ*scanBoxLen + scanBoxLen/2
                    f.write(f"\nH\t{boxX:.6f} {boxY:.6f} {boxZ:.6f}")
                for (boxIDX, boxIDY, boxIDZ) in bulkSets[i]:
                    boxX = minX - MIN_VAL_FROM_BOUND + boxIDX*scanBoxLen + scanBoxLen/2
                    boxY = minY - MIN_VAL_FROM_BOUND + boxIDY*scanBoxLen + scanBoxLen/2
                    boxZ = minZ - MIN_VAL_FROM_BOUND + boxIDZ*scanBoxLen + scanBoxLen/2
                    f.write(f"\nHe\t{boxX:.6f} {boxY:.6f} {boxZ:.6f}")

    return scaleChange, cntChange


def findSlope(scaleChange, countChange, NPname='', visReg=False, saveFig=False, showPlot=False):
    while np.nan in countChange:
        nanIdx = countChange.index(np.nan)
        del countChange[nanIdx]
        del scaleChange[nanIdx]
    firstPointIdx, lastPointIdx, r2score = countChange.count(countChange[0]), len(scaleChange), 0.0
    while r2score < R2_THRESHOLD:
        x, y = scaleChange[firstPointIdx:lastPointIdx], countChange[firstPointIdx:lastPointIdx]
        regModel = OLS(endog=y, exog=add_constant(x)).fit()
        r2score, boxCntDim, slopeConfInt = regModel.rsquared, regModel.params[1], \
                                             regModel.conf_int(alpha=ALPHA_CONF_INT)[1]
        yPred = regModel.predict()  # Returns ndarray, allowing subtraction later

        # Visualise the regression model
        if visReg:
            plt.clf()
            plt.scatter(x, y)
            plt.plot(x, yPred, label='OLS')
            if len(x) > 2:
                predOLS = regModel.get_prediction()
                lowCIvals, upCIvals = predOLS.summary_frame()['mean_ci_lower'], predOLS.summary_frame()['mean_ci_upper']
                plt.plot(x, upCIvals, 'r--')
                plt.plot(x, lowCIvals, 'r--')
            plt.xlabel('log(1/r)')
            plt.ylabel('log(N)')
            plt.title(f"{NPname} R2: {r2score:.3f}, D_Box: {boxCntDim:.3f}, {CONF_INT_PERC}% CI: [{slopeConfInt[0]:.3f}, "
                      f"{slopeConfInt[1]:.3f}]")
            if saveFig: plt.savefig(f"./boxCntDimFigs/{NPname}_BoxCntDim.png")
            if showPlot: plt.show()

        # Removal of next point (beware of weird behaviour in middle range, and arbitrary choice of R2 threshold)
        # lstSqErrs = np.subtract(y, yPred) ** 2
        # if len(y) % 2 == 0: lowBoundErrSum, upBoundErrSum = lstSqErrs[:len(y) // 2].sum(), lstSqErrs[len(y) // 2:].sum()
        # else: lowBoundErrSum, upBoundErrSum = lstSqErrs[:len(y) // 2].sum(), lstSqErrs[len(y) // 2 + 1:].sum()
        # if lowBoundErrSum > upBoundErrSum: firstPointIdx += 1
        # else: lastPointIdx -= 1
        firstPointIdx += 1
    return r2score, boxCntDim, slopeConfInt


@estDuration
def calcBoxCntDim(atomList, maxDimDiff, atomPosMinMax, findSurfOption='alphaShape'):
    # For running on NCI HPC facilities (Gadi) where data are stored
    minX, minY, minZ, maxX, maxY, maxZ = atomPosMinMax 
    findNN(atomList, minX, minY, minZ, maxX, maxY, maxZ)
    findSurf(atomList, option=findSurfOption, alpha=ALPHA)
    scaleChange, countChange = surfBoxCnt(atomList, maxDimDiff, minX, minY, minZ)
    return findSlope(scaleChange, countChange, NPname='', visReg=False, saveFig=False)


if __name__ == '__main__':
    verbose, calcAvgDuration, findSurfOption = True, False, 'alphaShape'
    testCases = ["COS1T000", "ICS1T000", "TOS1T000",
                 "CUS1T000", "CUS2T000", "CUS3T000",
                 "DHS1T000", "DHS2T000", "DHS3T000",
                 "OTS1T000", "OTS2T000", "OTS3T000",
                 "RDS1T000", "RDS2T000", "RDS3T000",
                 "THS1T000", "THS2T000", "THS3T000",
                 "OTS1T323", "OTS2T323", "OTS3T323",
                 "RDS1T323", "RDS2T323", "RDS3T323",
                 "THS1T323", "THS2T323", "THS3T323",
                 "OTS1T523", "OTS2T523", "OTS3T523",
                 "RDS1T523", "RDS2T523", "RDS3T523",
                 "THS1T523", "THS2T523", "THS3T523",
                 "disPd1", "disPd2", "disPd3"]  # DHS1T000 abnormal average bond length longer than others!!
    fracDimNPdict = {}
    for testCase in testCases:
        atomList, maxDimDiff, (minX, minY, minZ, maxX, maxY, maxZ) = readXYZ(f"../../../Data/NCPac/{testCase}.xyz")
        findNN(atomList, minX, minY, minZ, maxX, maxY, maxZ)
        findSurf(atomList, option=findSurfOption, alpha=ALPHA)
        (scaleChange, countChange), _ = surfBoxCnt(atomList, maxDimDiff, minX, minY, minZ, testCase, writeBox=True, verbose=False)
        r2score, boxCntDim, boxCntDimCI = findSlope(scaleChange, countChange, testCase, visReg=True, saveFig=False, showPlot=True)
        fracDimDict = {'boxCntDim': boxCntDim, 'lowCI': boxCntDimCI[0], 'upCI': boxCntDimCI[1], 'R2': r2score}
        fracDimNPdict[testCase] = fracDimDict
        print(f"Test case: {testCase}\t\tboxCntDim: {boxCntDim}")
    # fracDimDF = pd.DataFrame.from_dict(fracDimNPdict, orient='index')
    # fracDimDF['NPname'] = fracDimDF.index
    # fracDimDF['NPshape'] = fracDimDF['NPname'].apply(lambda x: x[:2])
    # fracDimDF['NPtemp'] = fracDimDF['NPname'].apply(lambda x: x[-3:])
    # TH = 4 {111}
    # CU = 6 {100}
    # OT = 8 {111}
    # RD = 12 {110}
    # TO = 8 {111}, 6 {100}
    # CO = 8 {111}, 6 {100}
    # DH = 10 {111}, 5 {100}  # Ino
    # IC = 20 {111}
    # numFaceOrderRank = {'TH': 1, 'CU': 2, 'OT': 3, 'RD': 4, 'TO': 5, 'CO': 6, 'DH': 7, 'IC': 8, 'di': 9}
    # fracDimDF['orderRank'] = fracDimDF['NPshape'].map(numFaceOrderRank)
    # fracDimDF.sort_values(by='orderRank')
    # fracDimIdealNPDF = fracDimDF[fracDimDF['NPtemp'] == '000']
    # fracDimIdealNPDF.groupby(by='NPshape').mean().sort_values(by='orderRank')
    # fracDimHeatedNPDF = fracDimDF[fracDimDF['NPtemp'] == '323']
    # fracDimHeatedNPDF.groupby(by='NPshape').mean().sort_values(by='orderRank')
    # with open('boxCntDimDict.pickle', 'wb') as f: pickle.dump(fracDimDF, f)
    # with open('boxCntDimDict.pickle', 'rb') as f: fracDimDF = pickle.load(f)
    # print(fracDimDF)

    if calcAvgDuration:
        avgDuration, repeatNum = 0, 10
        for i in range(repeatNum):
            _, duration = surfBoxCnt(atomList, maxDimDiff, minX, minY, minZ)
            avgDuration += duration
        print(f"Average duration: {avgDuration / repeatNum:.6f} s")

    if verbose:
        print(f"Change in\n\tScale: {scaleChange}\n\tCounts: {countChange}")
        print(f"Coefficient of determination (R2): {r2score:.3f}\
              \nEstimated box-counting dimension (D_Box): {boxCntDim:.3f}\
              \n{CONF_INT_PERC}% Confidence interval: [{boxCntDimCI[0]:.3f}, {boxCntDimCI[1]:.3f}]")