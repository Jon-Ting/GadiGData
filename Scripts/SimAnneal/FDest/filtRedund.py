# Goal: Filter redundant snapshots obtained from MD simulation of a given nanoparticle
# Author: Jonathan Yik Chang Ting
# Date: 27/10/2022
"""
Help on package filtRedund:

NAME
    filtRedund

DESCRIPTION
    Module to filter redundant snapshots from nanoparticle simulations
    ================================================
    Implemented:
    - Euclidean distance RMSD
    - Radial Distribution Function RMSD (didn't take element type into account, could use NCPac as well)
    - Q6Q6 average distribution
    - FD values

TO DO
    - graph embeddings (tough)
"""

import os
# import warnings
# import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from rdfpy import rdf
from scipy.stats import ks_2samp, cramervonmises_2samp, kruskal
from sklearn.metrics import mutual_info_score
from time import time
from FDestimation import *


# warnings.filterwarnings(action='ignore', category=RuntimeWarning)

# eleComb = 'AuPt'
# NP_DIRS_PATH = f"/scratch/q27/jt5911/SimAnneal/{eleComb}"

NP_DIRS_PATH = "./testCases/AuCo50RDCS_1957"
EUC_DIST_THRESH = 0.2
# Q6Q6_BOND_DIFF_THRESH = 0.001
BOX_CNT_DIM_DIFF_THRESH = 0.0005  # 0.0005 (visually undetectable) to XXX
RDF_DIFF_THRESH = 0.5  # 0.12 (visually undetectable) to 0.6 (1 atom moved)
RDF_DR = 0.001
R2_THRESHOLD = 0.99
ALPHA_CONF_INT = 0.05
CONF_INT_PERC = 1 - ALPHA_CONF_INT
PVAL_THRESH = 0.05


def estDuration(func):
    def wrap(*arg, **kwargs):
        start = time()
        result = func(*arg, **kwargs)
        end = time()
        duration = end - start
        # print(f"*** Function {func.__name__} duration: {duration:.6f} s\n")
        return result, duration
    return wrap


# def readQ6Q6(npName):
#     # Need to run NCPac.exe first, might not be feasible for large nanoparticles
#     q6q6bondList = []
#     with open(f"{NP_DIRS_PATH}/{npName}/ov_Q6Q6.xyz", 'r') as f:
#         for (i, line) in enumerate(f):
#             if i < 2: continue
#             q6q6bondNum = float(line.split()[-1])
#             q6q6bondList.append(q6q6bondNum)
#     return q6q6bondList


def alignNP(P, Q):
    '''
    Align P with Q (reference structure) using Kabsch algorithm
    Algorithm explanation: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Reference code: https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
    '''
    # Translation of centroids of both structures to system origin
    P_cent, Q_cent = P - P.mean(axis=0), Q - Q.mean(axis=0)
    C = np.dot(P_cent.T, Q_cent)  # Covariance matrix
    V, S, W = np.linalg.svd(C)  # Singular value decomposition
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0  # Proper/improper rotation
    if d: S[-1], V[:, -1], = -S[-1], -V[:, -1] # Antialigns of the last singular vector
    U = np.dot(V, W)  # Rotation matrix
    P_align = np.dot(P_cent, U) + P.mean(axis=0)
    return P_align
    

@estDuration
def calcEucRMSD(coordList1, coordList2):
    eucRMSD = np.sqrt(np.mean(np.square(np.array(coordList1) - np.array(coordList2))))
    return eucRMSD


def calcMI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mutualInf = mutual_info_score(None, None, contingency=c_xy)
    return mutualInf


@estDuration
def calcRDFdiff(coordList1, coordList2, showPlot=False):
    g_r1, radii1 = rdf(coordList1, dr=RDF_DR)
    g_r2, radii2 = rdf(coordList2, dr=RDF_DR)

    while np.isnan(g_r1).any() or np.isnan(g_r2).any():
        g_r1, g_r2 = g_r1[:-1], g_r2[:-1]
    dStat, pValD = ks_2samp(g_r1, g_r2, alternative='two-sided')  # 0 (same) < D < 1 (different)
    hStat, pValH = kruskal(g_r1, g_r2)  # H statistics corrected for ties, significant result means > 1 sample stochastically dominates one other sample
    # cStat, pValC = cramervonmises_2samp(g_r1, g_r2, method='asymptotic')  # 'exact' option suffers from combinatorial explosion, inappropriate for unequal number of sample

    if len(g_r1) != len(g_r2): 
        # print(f"Length: {len(g_r1)}, {len(g_r2)}")
        # print(f"Start radii: {radii1[:5]}, {radii2[:5]}")
        # print(f"End radii: {radii1[-5:]}, {radii2[-5:]}")
        if len(g_r1) < len(g_r2): g_r2 = g_r2[:len(g_r1)]
        else: g_r1 = g_r1[:len(g_r2)]
    # if showPlot:
    #     plt.plot(radii1, g_r1);
    #     plt.plot(radii2, g_r2);
    #     plt.xlabel('r');
    #     plt.ylabel('g(r)');
    #     plt.legend(['NP1', 'NP2']);
    #     plt.show();
    mutualInf = calcMI(g_r1, g_r2, bins=100)
    return dStat, pValD, hStat, pValH, mutualInf # np.sqrt(np.sum(np.square(g_r1 - g_r2)))


# def calcQ6Q6bondDiff(npName1, npName2):
#     q6q6bonds1, q6q6bonds2 = readQ6Q6(npName1), readQ6Q6(npName2) 
#     q6q6Diff = np.sqrt(np.sum(np.square(np.array(q6q6bonds1) - np.array(q6q6bonds2))))
#     return q6q6Diff


def runFilter():
    foundRedund = False
    NPfiltIdxs = set()
    sortedDirs = natsorted(os.listdir(NP_DIRS_PATH))
    eucTs, rdfTs, q6q6Ts, bcdTs = [], [], [], []
    for (i, npName1) in enumerate(sortedDirs):
        if i > 30: break
        if i not in NPfiltIdxs:
            filePath1 = f"{NP_DIRS_PATH}/{npName1}"
            atomList1, maxDimDiff1, atomPosMinMax1 = readXYZ(filePath1)
            coordList1 = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList1])
            (r2dim1, boxCntDim1, CIdim1), bcdT1 = calcBoxCntDim(atomList1, maxDimDiff1, atomPosMinMax1)

        j = j + 1 if foundRedund else i + 1
        if j >= len(sortedDirs): break
        npName2 = sortedDirs[j]
        # if j != i+1: continue  # j!=i+1 (consecutive) / j>=i (all other)
        print(f"Comparing {npName1} with {npName2}...")

        filePath2 = f"{NP_DIRS_PATH}/{npName2}"
        atomList2, maxDimDiff2, atomPosMinMax2 = readXYZ(filePath2)
        coordList2 = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList2])
        coordList2 = alignNP(coordList2, coordList1)
        (r2dim2, boxCntDim2, CIdim2), bcdT2 = calcBoxCntDim(atomList2, maxDimDiff2, atomPosMinMax2)

        # Compute difference metrics
        eucRMSD, eucT = calcEucRMSD(coordList1, coordList2)
        (rdfStatD, rdfPValD, rdfStatH, rdfPValH, rdfMI), rdfT = calcRDFdiff(coordList1, coordList2)
        # q6q6Diff = calcQ6Q6bondDiff(npName1, npName2)
        boxCntDimDiff = abs(boxCntDim2 - boxCntDim1)
        eucTs.append(eucT)
        rdfTs.append(rdfT)
        bcdTs.append(bcdT1 + bcdT2)
        print(f"\teucRMSD: {eucRMSD:.5f}")
        # print(f"\tq6q6Diff: {q6q6Diff}")
        print(f"\tboxCntDimDiff: {boxCntDim1:.6f} - {boxCntDim2:.6f} = {boxCntDimDiff:.6f}")
        print(f"\t2-sample Kolmogorov-Smirnov test statistic: {rdfStatD:.5f}; p-value: {rdfPValD:.6f}")
        print(f"\tKruskal-Wallis H-test statistic: {rdfStatH:.5f}; p-value: {rdfPValH:.6f}")
        print(f"\tMutual information : {rdfMI:.3f}")

        if eucRMSD < EUC_DIST_THRESH and boxCntDimDiff < BOX_CNT_DIM_DIFF_THRESH and rdfPValD > PVAL_THRESH and rdfPValH > PVAL_THRESH and False:
            print('Found a redundant frame:')
            NPfiltIdxs.add(j) 
            foundRedund = True
            continue
        foundRedund = False
    print(f"Average duration:\n  RMSD {sum(eucTs)/len(eucTs):.6f} s\n  RDF {sum(rdfTs)/len(rdfTs):.6f} s\n  BCDim {sum(bcdTs)/len(bcdTs):.6f}")
    return NPfiltIdxs


if __name__ == '__main__':
    NPfiltIdxs = runFilter()
    print(NPfiltIdxs)
