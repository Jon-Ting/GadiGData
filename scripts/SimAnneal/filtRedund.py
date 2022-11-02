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

import csv
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.api import OLS, add_constant
from natsort import natsorted, ns
from rdfpy import rdf


# warnings.filterwarnings(action='ignore', category=RuntimeWarning)

eleComb = 'AuPt'
NP_DIRS_PATH = f"/scratch/q27/jt5911/SimAnneal/{eleComb}"
NP_DIRS_PATH = "./testDir2"
EUC_DIST_THRESH = 0.2
Q6Q6_BOND_DIFF_THRESH = 0.001
BOX_CNT_DIM_DIFF_THRESH = 0.0005  # 0.0005 (visually undetectable) to XXX
RDF_DIFF_THRESH = 0.5  # 0.12 (visually undetectable) to 0.6 (1 atom moved)
RDF_DR = 0.05
R2_THRESHOLD = 0.99
ALPHA_CONF_INT = 0.05
CONF_INT_PERC = 1 - ALPHA_CONF_INT

def readXYZ(npName):
    coordList = []
    with open(f"{NP_DIRS_PATH}/{npName}/{npName}.xyz", 'r') as f:
        for (i, line) in enumerate(f):
            if i < 2: continue
            atomCoord = [float(coord) for coord in line.split()[1:]]
            coordList.append(atomCoord)
    return coordList


def readQ6Q6(npName):
    q6q6bondList = []
    with open(f"{NP_DIRS_PATH}/{npName}/ov_Q6Q6.xyz", 'r') as f:
        for (i, line) in enumerate(f):
            if i < 2: continue
            q6q6bondNum = float(line.split()[-1])
            q6q6bondList.append(q6q6bondNum)
    return q6q6bondList


def calcEucDist(coordList1, coordList2):
    eucDist = np.sqrt(np.sum(np.square(np.array(coordList1) - np.array(coordList2))))
    return eucDist


def calcRDFdiff(coordList1, coordList2, showPlot=False):
    g_r1, radii1 = rdf(coordList1, dr=RDF_DR)
    g_r2, radii2 = rdf(coordList2, dr=RDF_DR)
    if len(g_r1) != len(g_r2): 
        # print(f"Length: {len(g_r1)}, {len(g_r2)}")
        # print(f"Start radii: {radii1[:5]}, {radii2[:5]}")
        # print(f"End radii: {radii1[-5:]}, {radii2[-5:]}")
        if len(g_r1) < len(g_r2): g_r2 = g_r2[:len(g_r1)]
        else: g_r1 = g_r1[:len(g_r2)]
    if showPlot:
        plt.plot(radii1, g_r1);
        plt.plot(radii2, g_r2);
        plt.xlabel('r');
        plt.ylabel('g(r)');
        plt.legend(['NP1', 'NP2']);
        plt.show();
    return np.sqrt(np.sum(np.square(g_r1 - g_r2)))


def calcQ6Q6bondDiff(npName1, npName2):
    q6q6bonds1, q6q6bonds2 = readQ6Q6(npName1), readQ6Q6(npName2) 
    q6q6Diff = np.sqrt(np.sum(np.square(np.array(q6q6bonds1) - np.array(q6q6bonds2))))
    return q6q6Diff


def calcNPSurfBoxCntDim(npName, saveFig=False):
    with open(f"{NP_DIRS_PATH}/{npName}/od_FRADIM_NP_SURF.csv") as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for (i, row) in enumerate(csvReader):
            if i == 0: continue  # Skip the header
            elif i == 1: scaleChange = np.array([float(logScale) for logScale in row[1:]]).reshape(-1,1)  # Get log(1/r)
            elif i == 2: 
                minCountDuplicates = row.count(row[1])  # Number of duplicates of the minimum count
                firstPointIdx, lastPointIdx = minCountDuplicates + 1, None  # Remove all duplicates (zeros/inf)
                countChange = np.array([float(logCounts) for logCounts in row[1:]]).reshape(-1,1)[firstPointIdx:lastPointIdx]
                scaleChange = scaleChange[firstPointIdx:lastPointIdx]
            elif i == 3:  # Only for surfaces
                minCountDuplicates = row.count(row[1])
            else: print("Unexpected line!")

    # print(f"  Fitting linear line to estimate fractal dimension of nanoparticle represented as surface:")
    r2score = 0.0
    firstPointIdx, lastPointIdx = 0, len(scaleChange) - 1
    while r2score < R2_THRESHOLD:
        x, y = scaleChange[firstPointIdx:lastPointIdx], countChange[firstPointIdx:lastPointIdx]
        regModel = OLS(endog=y, exog=add_constant(x)).fit()
        r2score, boxCntDim, slopeConfInt = regModel.rsquared, regModel.params[1], regModel.conf_int(alpha=ALPHA_CONF_INT)[1]
        # print(f"    Coefficient of determination (R2): {r2score:.3f}\
        #       \n    Estimated box-counting dimension (D_Box): {boxCntDim:.3f}\
        #       \n    {CONF_INT_PERC}% Confidence interval: [{slopeConfInt[0]:.3f}, {slopeConfInt[1]:.3f}]")

        # Visualise the regression model
        yPred = regModel.predict().reshape(-1, 1)
        if saveFig:
            plt.scatter(x, y);
            plt.plot(x, yPred, label='OLS');
            if len(x) > 2:
                predOLS = regModel.get_prediction()
                lowerCIvals, upperCIvals = predOLS.summary_frame()['mean_ci_lower'], predOLS.summary_frame()['mean_ci_upper']
                plt.plot(x, upperCIvals, 'r--');
                plt.plot(x, lowerCIvals, 'r--');
            plt.xlabel('log(1/r)');
            plt.ylabel('log(N)');
            plt.title(f"R2: {r2score:.3f}, D_Box: {boxCountDim:.3f}, {CONF_INT_PERC}% CI: [{slopeConfInt[0]:.3f}, {slopeConfInt[1]:.3f}]");
            # plt.savefig(f"{NP_DIRS_PATH[:-4]}/figures/", dpi=300);
        
        # Decide the next point to remove
        leastSquareErrs = (y - yPred)**2
        if len(y) % 2 == 0: lowerBoundErrSum, upperBoundErrSum = leastSquareErrs[:len(y)//2].sum(), leastSquareErrs[len(y)//2:].sum()
        else: lowerBoundErrSum, upperBoundErrSum = leastSquareErrs[:len(y)//2].sum(), leastSquareErrs[len(y)//2 + 1:].sum()
        if lowerBoundErrSum > upperBoundErrSum: firstPointIdx += 1
        else: lastPointIdx -= 1
    
    return boxCntDim  #, r2score, slopeConfInt[0], slopeConfInt[1]


def main():
    NPfiltIdxs = []
    for (i, npName1) in enumerate(natsorted(os.listdir(NP_DIRS_PATH))):
        coordList1 = readXYZ(npName1)

        boxCntDim1 = calcNPSurfBoxCntDim(npName1, saveFig=False)
        for (j, npName2) in enumerate(natsorted(os.listdir(NP_DIRS_PATH))):
            if j != i + 1: continue  # j >= i to compare with all other frames
            print(f"Comparing {npName1} with {npName2}...")
            coordList2 = readXYZ(npName2)

            # Compute difference metrics
            eucDist = calcEucDist(coordList1, coordList2)
            rdfDiff = calcRDFdiff(coordList1, coordList2)
            q6q6Diff = calcQ6Q6bondDiff(npName1, npName2)
            boxCntDim2 = calcNPSurfBoxCntDim(npName2, saveFig=False)
            boxCntDimDiff = abs(boxCntDim2 - boxCntDim1)

            if eucDist < EUC_DIST_THRESH and rdfDiff < RDF_DIFF_THRESH and q6q6Diff < Q6Q6_BOND_DIFF_THRESH and boxCntDimDiff < BOX_CNT_DIM_DIFF_THRESH:
                print('Found a redundant frame:')
                print(f"\teucDist: {eucDist}")
                print(f"\trdfDiff: {rdfDiff}")
                print(f"\tq6q6Diff: {q6q6Diff}")
                print(f"\tboxCntDimDiff: {boxCntDimDiff}")
                NPfiltIdxs.append(j) 
    return NPfiltIdxs


if __name__ == '__main__':
    NPfiltIdxs = main()
    print(NPfiltIdxs)
