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
    - Radial Distribution Function Mutual Information (didn't take element type into account, could use NCPac as well)
    - Q6Q6 average distribution (Requires NCPac to be run first, which defeats the purpose of filtering redundant frames prior to feature extraction, thus omitted)
    - FD values

TODO
    - Explore the potential of graph embeddings for this purpose
"""

import os
import pickle
import warnings
from FDestimation import *  # Includes numpy and time
from natsort import natsorted
from pyinform.mutualinfo import mutual_info
from rdfpy import rdf
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp, wasserstein_distance, gaussian_kde
from scipy.special import rel_entr
from sklearn.feature_selection import mutual_info_regression


# Control printout to ease debugging, to be commented out eventually
warnings.filterwarnings(action='ignore', category=RuntimeWarning)  # Addresses warnings from 2-sample Kolmogorov-Smirnov test regarding switching to asymp from exact for p-value calculation
warnings.filterwarnings(action='ignore', category=UserWarning)  # Addresses warnings from k-sample Anderson-Darling test regarding flooring/capping p-value

NUM_FRAME_PER_NP = 101
NUM_UNIQ_THRESH = 5
PVAL_THRESH = 0.05
RDF_RCUTOFF_MULTIPLIER = 0.9  # Minimise noise at edge particles
RDF_NUM_SAMPLES = 1000  # Balance between computational cost and statistical test power

# Fine-tuned based mainly on AuPd20CUL10, AuPd30ICRCCS_923, CoPd40TOL12_4033
EUC_DIST_THRESH = 0.05  # 0.032-0.063
BOX_CNT_DIM_DIFF_THRESH = 0.01  # 0.0061-0.0229
MI_THRESH = 1.0  # 0.86-1.29
WS_THRESH = 0.0001  # 0.00004-0.00020
HL_THRESH = 0.08  # 0.06-0.09
KL_THRESH = 0.25  # 0.22-0.27
JS_THRESH = 0.005  # 0.003-0.008


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


def getCDF(p, covFactor=0.1):
    cdf = gaussian_kde(p)
    cdf.covariance_factor = lambda: covFactor
    cdf._compute_covariance()
    return cdf


def calcBCdist(p, q):
    # https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py
    pCDF, qCDF = getCDF(p), getCDF(q)
    minVal, maxVal, numSteps = min(np.concatenate((p, q))), max(np.concatenate((p, q))), 100
    xs = np.linspace(minVal, maxVal, numSteps)
    return sum([np.sqrt(pCDF(x)[0]*qCDF(x)[0])*(maxVal-minVal)/numSteps for x in xs])


def calcKLdiv(p, q):
    return np.sum(rel_entr(p+10e-128, q+10e-128))  # rel_entr == p*np.log(p/q)


@estDuration
def calcRDFdiff(coordList1, coordList2, minMaxDimDiff, timeBenchmark=False, showPlot=False):
    rdfInt = round(minMaxDimDiff/2 * RDF_RCUTOFF_MULTIPLIER / RDF_NUM_SAMPLES, 3)
    gR1, radii1 = rdf(coordList1, dr=rdfInt)
    gR2, radii2 = rdf(coordList2, dr=rdfInt)
    if len(np.unique(gR1)) == 1 or len(np.unique(gR2)) == 1: return tuple([np.nan] * 12)
    # while np.isnan(gR1).any() or np.isnan(gR2).any(): gR1, gR2 = gR1[:-1], gR2[:-1]  # Not needed anymore, problem circumvented by adding a line to rdfpy
    A = anderson_ksamp([gR1, gR2], midrank=True)  # H0: F=G
    D = ks_2samp(gR1, gR2, alternative='two-sided')  # 0 (same) < D < 1 (different)
    W = cramervonmises_2samp(gR1, gR2, method='asymptotic')  # 'exact' option suffers from combinatorial explosion
    # if showPlot:
    #     import matplotlib.pyplot as plt
    #     plt.plot(radii1, gR1);
    #     plt.plot(radii2, gR2);
    #     plt.xlabel('r');
    #     plt.ylabel('g(r)');
    #     plt.legend(['NP1', 'NP2']);
    #     plt.show();  # plt.savefig('test.png');
    if len(gR1) != len(gR2):  # Cut the last points from the slightly longer g_r
        if len(gR1) < len(gR2): gR2 = gR2[:len(gR1)]
        else: gR1 = gR1[:len(gR2)]
    gR1norm, gR2norm = gR1/gR1.sum(), gR2/gR2.sum()
    wsDist = wasserstein_distance(gR1norm, gR2norm, u_weights=None, v_weights=None)  # Wasserstein distance
    hlDist = np.sqrt(np.sum((np.sqrt(gR1norm) - np.sqrt(gR2norm)) ** 2)) / np.sqrt(2)  # Hellinger distance  
    klDiv = calcKLdiv(gR1norm, gR2norm)  # Kullback-Leibler divergence, asymmetrical
    m = 0.5*(gR1norm+gR2norm)
    jsDiv = 0.5*calcKLdiv(gR1norm, m) + 0.5*calcKLdiv(gR2norm, m)  # Jensen-Shannon divergence, symmetrical
    if timeBenchmark:
        bcDist = calcBCdist(gR1norm, gR2norm) # Bhattacharyya distance
        miScore = mutual_info(gR1, gR2)  # Mutual information for time series (though it isn't here) due to correlated nature (joint/pair distribution needed, not just marginal distributions)
    else: bcDist, miScore = None, None
    return A.statistic, A.pvalue, D.statistic, D.pvalue, W.statistic, W.pvalue, wsDist, hlDist, klDiv, jsDiv, bcDist, miScore  #, np.sqrt(np.sum(np.square(gR1 - gR2)))


# def readQ6Q6(npName):
#     # Need to run NCPac.exe first, might not be feasible for large nanoparticles
#     q6q6bondList = []
#     with open(f"{NP_DIRS_PATH}/{npName}/ov_Q6Q6.xyz", 'r') as f:
#         for (i, line) in enumerate(f):
#             if i < 2: continue
#             q6q6bondNum = float(line.split()[-1])
#             q6q6bondList.append(q6q6bondNum)
#     return q6q6bondList


def runFilter(targetDir, startNPidx=0, timeBenchmark=False, verbose=False):
    eleComb = targetDir.split('/')[-1]
    if verbose: print(f"Filtering redundant frames from {eleComb} nanoparticles simulations...")
    cntUnique, foundRedund, calcBL = 0, False, True
    NPfiltNames = pickle.load(open(f"{eleComb}_redundNPnames.pickle", 'rb')) if os.path.exists(f"{eleComb}_redundNPnames.pickle") else set()
    sortedDirs = natsorted(os.listdir(targetDir))
    numAtom1s, numAtom2s, numSurfAtom1s, numSurfAtom2s, eucTs, rdfTs, bcdTs, bcdT1s, bcdT2s, blcnTs = [], [], [], [], [], [], [], [], [], []
    if timeBenchmark: NPsimilarityDict, calcBL = {}, False

    for (i, npName1) in enumerate(sortedDirs):
        if not timeBenchmark:
            if not os.path.isdir(f"{targetDir}/{npName1}"): 
                if verbose: print(f"  ***SKIP*** {npName1} is not a directory, skipping...")
                continue
            if int(npName1) < startNPidx: continue # Start from specified index in case some of them has been processed
            # if int(npName1) % (NUM_FRAME_PER_NP * NUM_UNIQ_THRESH) == 0:  # Update saved list containing redundant frames
            #     with open(f"{eleComb}_redundNPnames.pickle", 'wb') as f: pickle.dump(NPfiltNames, f)
        
            if i != 0 and (int(npName1)+1) % NUM_FRAME_PER_NP == 0: 
                if verbose: print(f"  ***SKIP*** Last frame {npName1}, skipping comparison with first frame of next NP...")
                cntUnique = 0
                continue
            if cntUnique > NUM_UNIQ_THRESH:  # The rest of NPs are likely unique, skip to next NP to save time 
                if int(npName1) % NUM_FRAME_PER_NP != 0: continue

        # Compute similarity measures for 1st NP
        filePath1 = f"{targetDir}/{npName1}" if timeBenchmark else f"{targetDir}/{npName1}/{npName1}.xyz"
        atomList1, maxDimDiff1, eleSet1, atomPosMinMax1 = readXYZ(filePath1)
        coordList1 = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList1])
        (numSurfAtom1, r2dim1, boxCntDim1, CIdim1, avgBondLen1), bcdT1 = calcBoxCntDim(atomList1, maxDimDiff1, eleSet1, atomPosMinMax1, calcBL=calcBL)
        if timeBenchmark: (avgBondLen1, coordNum1), blcnT1 = calcBLCN(atomList1, eleSet1, atomPosMinMax1)

        # Compare with consecutive frame
        j = i + 1
        if j >= len(sortedDirs): break
        npName2 = sortedDirs[j]
        print(f"  Comparing {npName1} with {npName2}...")

        # Compute similarity measures for 2nd NP
        if not timeBenchmark:
            if not os.path.isdir(f"{targetDir}/{npName2}"): 
                if verbose: print(f"  ***SKIP*** {npName2} is not a directory, skipping...")
                continue
        filePath2 = f"{targetDir}/{npName2}" if timeBenchmark else f"{targetDir}/{npName2}/{npName2}.xyz"
        atomList2, maxDimDiff2, eleSet2, atomPosMinMax2 = readXYZ(filePath2)
        coordList2 = np.array([[atom.X, atom.Y, atom.Z] for atom in atomList2])
        coordList2 = alignNP(coordList2, coordList1)  # Doesn't impact FD estimation
        (numSurfAtom2, r2dim2, boxCntDim2, CIdim2, avgBondLen2), bcdT2 = calcBoxCntDim(atomList2, maxDimDiff2, eleSet2, atomPosMinMax2, calcBL=calcBL)
        if timeBenchmark: (avgBondLen2, coordNum2), blcnT2 = calcBLCN(atomList2, eleSet2, atomPosMinMax2)

        # Compute and report all similarity measures
        eucRMSD, eucT = calcEucRMSD(coordList1, coordList2)
        (rdfMeasures), rdfT = calcRDFdiff(coordList1, coordList2, min(maxDimDiff1, maxDimDiff2), timeBenchmark=timeBenchmark)
        rdfStatA, rdfPValA, rdfStatD, rdfPValD, rdfStatW, rdfPValW, rdfWS, rdfHL, rdfKL, rdfJS, rdfBC, rdfMI = rdfMeasures
        boxCntDimDiff = abs(boxCntDim2 - boxCntDim1)
        avgBondLenMI = mutual_info_regression(avgBondLen1.reshape((-1, 1)), avgBondLen2, random_state=777)[0]
        numAtom1s.append(len(coordList1))
        numAtom2s.append(len(coordList2))
        numSurfAtom1s.append(numSurfAtom1)
        numSurfAtom2s.append(numSurfAtom2)
        eucTs.append(eucT)
        rdfTs.append(rdfT)
        bcdTs.append(bcdT1 + bcdT2)
        bcdT1s.append(bcdT1)
        bcdT2s.append(bcdT2)
        if verbose:
            print(f"    eucRMSD: {eucRMSD:.3f}")
            print(f"    boxCntDimDiff: |{boxCntDim1:.4f} - {boxCntDim2:.4f}| = {boxCntDimDiff:.4f}")
            print(f"    avgBondLen Mutual information : {avgBondLenMI:.2f}")
            print(f"    RDF 2-sample Anderson-Darling test statistic: {rdfStatA:.4f}; p-value: {rdfPValA:.3f}")
            print(f"    RDF 2-sample Kolmogorov-Smirnov test statistic: {rdfStatD:.4f}; p-value: {rdfPValD:.3f}")
            print(f"    RDF 2-sample Cramer-von Mises test statistic: {rdfStatW:.4f}; p-value: {rdfPValW:.3f}")
            print(f"    RDF Wasserstein distance : {rdfWS:.7f}")
            print(f"    RDF Hellinger distance : {rdfHL:.3f}")
            print(f"    RDF Kullback-Leibler divergence : {rdfKL:.2f}")
            print(f"    RDF Jensen-Shannon divergence : {rdfJS:.5f}")

        if timeBenchmark:
            coordNumMI = mutual_info_regression(coordNum1.reshape((-1, 1)), coordNum2, random_state=777)[0]
            blcnTs.append(blcnT1 + blcnT2)
            if verbose:
                print(f"    RDF Mutual information : {rdfMI:.2f}")
                print(f"    RDF Bhattacharyya distance : {rdfBC:.2f}")
                print(f"    coordNum Mutual information : {coordNumMI:.2f}")
            similarityDict = {'numAtom1s': numAtom1s, 'numAtom2s': numAtom2s, 'numSurfAtom1s': numSurfAtom1s, 'numSurfAtom2s': numSurfAtom2s, 'eucRMSD': eucRMSD, 'boxCntDimDiff': boxCntDimDiff, 'rdfStatA': rdfStatA, 'rdfPValA': rdfPValA, 'rdfStatD': rdfStatD, 'rdfPValD': rdfPValD, 'rdfStatW': rdfStatW, 'rdfPValW': rdfPValW, 'rdfBC': rdfBC, 'rdfWS': rdfWS, 'rdfHL': rdfHL, 'rdfKL': rdfKL, 'rdfJS': rdfJS, 'rdfMI': rdfMI, 'avgBondLenMI': avgBondLenMI, 'coordNumMI': coordNumMI, 'eucTs': eucTs, 'rdfTs': rdfTs, 'bcdTs': bcdTs, 'bcdT1s': bcdT1s, 'bcdT2s': bcdT2s, 'blcnTs': blcnTs}
            NPsimilarityDict[NP_NAME] = similarityDict

        # Determine if 1st NP is redundant
        if rdfMeasures[0] == np.nan:  #TODO: come up with better test combination 
            redundCond = eucRMSD < EUC_DIST_THRESH and boxCntDimDiff < BOX_CNT_DIM_DIFF_THRESH and avgBondLenMI > MI_THRESH
        else: redundCond = eucRMSD < EUC_DIST_THRESH and boxCntDimDiff < BOX_CNT_DIM_DIFF_THRESH and avgBondLenMI > MI_THRESH and rdfPValA > PVAL_THRESH and rdfPValD > PVAL_THRESH and rdfPValW > PVAL_THRESH and rdfWS < WS_THRESH and rdfHL < HL_THRESH and rdfKL < KL_THRESH and rdfJS < JS_THRESH
        if redundCond:
            if verbose: print(f"      FOUND: {npName1} redundant!!!")
            NPfiltNames.add(npName1)
            foundRedund, cntUnique = True, 0
            continue
        foundRedund, cntUnique = False, cntUnique + 1

    if timeBenchmark:
        print(f"***Average duration (s):\n  RMSD {sum(eucTs)/len(eucTs):.6f}\n  RDF {sum(rdfTs)/len(rdfTs):.6f}\n  BCDim {sum(bcdTs)/len(bcdTs):.6f}")
        filtRedundDF = pd.DataFrame.from_dict(NPsimilarityDict, orient='index')
        filtRedundDF['NPname'] = filtRedundDF.index
        with open(f"{NP_NAME}_filtRedund.pickle", 'wb') as f: pickle.dump(filtRedundDF, f)

    # with open(f"{eleComb}_redundNPnames.pickle", 'wb') as f: pickle.dump(NPfiltNames, f)
    if verbose: print(f"Redundant frames found: {NPfiltNames}")
    return NPfiltNames


if __name__ == '__main__':
    timeBenchmark, rmRedund, reorderCSV, reorderXYZ = True, False, False, False
    if timeBenchmark:
        import sys
        NP_NAME = sys.argv[1] 
        NP_DIRS_PATH = f"./testCasesFiltRedund/{NP_NAME}"
    else:
        eleComb = 'AuPd'  # IMPORTANT VARIABLE
        NP_DIRS_PATH = f"/scratch/q27/jt5911/{eleComb}"
    startNPidx = 0  # IMPORTANT VARIABLE
    NPfiltNames = runFilter(targetDir=NP_DIRS_PATH, startNPidx=startNPidx, timeBenchmark=timeBenchmark, verbose=True)
    # NPfiltNames = pickle.load(open(f"{eleComb}_redundNPnames.pickle", 'rb'))

    if rmRedund: 
        from shutil import move
        REDUND_DIR_PATH = f"{NP_DIRS_PATH}/REDUNDANT"
        if not os.path.isdir(REDUND_DIR_PATH): 
            os.mkdir(REDUND_DIR_PATH)
            for npName in NPfiltNames: move(f"{NP_DIRS_PATH}/{npName}", f"{REDUND_DIR_PATH}")
    if reorderCSV:
        with open(f"{NP_DIRS_PATH}_nanoparticle_data/{eleComb}_allNP_allFeat.csv", 'r') as f1:
            with open(f"{NP_DIRS_PATH}_nanoparticle_data/{eleComb}_allFeat.csv", 'w') as f2:
                for (i, line) in enumerate(f1):
                    if line.split(',')[0] in NPfiltNames: continue
                    f2.write(line)
    if reorderXYZ:
        from shutil import copy
        NPnames = os.listdir(NP_DIRS_PATH)
        for (i, NPname) in enumerate(NPnames):
            copy(f"{NP_DIRS_PATH}/{NPname}/{NPname}.xyz", f"{NP_DIRS_PATH}TMP/DAP_{eleComb}/{str(i+1).zfill(6)}.xyz")

    print('ALL DONE!')
