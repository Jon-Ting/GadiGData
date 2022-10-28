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

TO DO
    - Euclidean distance
    - FD values
    - RDF average
    - Q6Q6 average distribution
    - mutual information score (of?)
    - graph embeddings (tough)
"""


import os
import numpy as np
from natsort import natsorted, ns


NPdirPath = './testDir'
eucDistThresh = 0.1
boxCntDimThresh = 0.001



def calcEucDist(xyzNP1, xyzNP2):
    coordList1, coordList2 = [], []
    with open(f"{NPdirPath}/{xyzNP1}", 'r') as f1:
        for (i, lines) in enumerate(f1.readlines()):
            if i < 2: continue
            atomCoord = [float(coord) for coord in lines.split()[1:]]
            coordList1.append(atomCoord)
    with open(f"{NPdirPath}/{xyzNP2}", 'r') as f2:
        coordList2 = []
        for (i, lines) in enumerate(f2.readlines()):
            if i < 2: continue
            atomCoord = [float(coord) for coord in lines.split()[1:]]
            coordList2.append(atomCoord)
    return np.sqrt(np.sum(np.square(np.array(coordList1) - np.array(coordList2))))


def calcNPBoxCntDim(xyzNP):
    boxCntDim = 0
    # Run NCPac.exe to get FD
    return boxCntDim


def main():
    NPfiltIdxs = []
    for (i, xyzFrame1) in enumerate(natsorted(os.listdir(NPdirPath))):
        boxCntDim1 = calcNPBoxCntDim(xyzFrame1)
        for (j, xyzFrame2) in enumerate(natsorted(os.listdir(NPdirPath))):
            if j != i + 1: continue  # j >= i to compare with all other frames
            eucDist = calcEucDist(xyzFrame1, xyzFrame2)
            print(eucDist)
            boxCntDim2 = calcNPBoxCntDim(xyzFrame2)
            boxCntDimDiff = boxCntDim2 - boxCntDim1 
            if eucDist < eucDistThresh and boxCntDimDiff < boxCntDimThresh:
               NPfiltIdxs.append(j) 
    return NPfiltIdxs


if __name__ == '__main__':
    main()
