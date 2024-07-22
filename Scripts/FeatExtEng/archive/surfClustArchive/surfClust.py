#!/usr/bin/env python
import os
import random
import time
import sys
import warnings
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import iqr, skew
import seaborn as sns
import sklearn

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MeanShift, SpectralClustering
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import ils.dirichletMixtures as dm
import ils.agglomerativeClustering as hc
import ils.utils as ut
import ils.cfsdp as cfs
from ils.basic_ils import ILS

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from itertools import combinations
from KDEpy import NaiveKDE, FFTKDE
from scipy.integrate import romb, simps
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks_cwt
from scipy.special import rel_entr

from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import metrics
import scipy.spatial.distance as dist
from multiprocessing import Pool


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')
sns.set_context('paper')
sns.set(color_codes=True)
sns.set(font_scale=1.2)
warnings.filterwarnings('ignore')
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
#rcParams['figure.figsize'] = [8, 5]
#rcParams['figure.dpi'] = 80
rcParams['figure.autolayout'] = True
rcParams['font.style'] = 'normal'
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
randomSeed = 777
# np.set_printoptions(precision=4, suppress=True)


# Print version for reproducibility
#print(f"numpy version {np.__version__}")
#print(f"pandas version {pd.__version__}")
#print(f"seaborn version {sns.__version__}")
#print(f"sklearn version {sklearn.__version__}")


# ## Load Data
npName, startID, endID, clustMethod = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
testCases = [ str(i).zfill(4) for i in range(2501)]
with open(f'pickleFiles/biFeat_dfAlls_{npName}.pickle', 'rb') as f: dfAlls = pickle.load(f)
with open(f'pickleFiles/biFeat_dfNoLVHCs_laplacianScore80_{npName}.pickle', 'rb') as f: dfNoLVHCs = pickle.load(f)
with open(f'pickleFiles/biFeat_dfScaledNoLVHCs_laplacianScore80_{npName}.pickle', 'rb') as f: dfScaledNoLVHCs = pickle.load(f)
#varThreshIdx, corrThreshIdx = 0, 1  # Investigating data retaining only features with variance > 0.01 and correlation < 0.95 with another feature
#dfNoLVHCs, dfScaledNoLVHCs = dfNoLVHCs[varThreshIdx][corrThreshIdx], dfScaledNoLVHCs[varThreshIdx][corrThreshIdx]


# Features that do not satisfy rotational invariance are excluded
# spaceList = ['x', 'y', 'z']
blList = ['bl1min', 'bl1num', 'bl2avg', 'bl2min', 'bl2num', 'bl3avg', 'bl3min', 'bl3num', 'bl4avg', 'bl4max', 'bl4min']
ba1List = ['ba1avg', 'ba1max', 'ba11avg', 'ba11min', 'ba12avg', 'ba12min', 'ba12num','ba13avg', 'ba13min', 'ba13num',
           'ba14avg', 'ba14min','ba15avg', 'ba15min', 'ba16avg', 'ba16min', 'ba16num','ba17avg', 'ba17min', 'ba17num',
           'ba18avg', 'ba18min']
ba2List = ['ba21avg', 'ba21min', 'ba22avg', 'ba22max', 'ba22min', 'ba23avg', 'ba23min',
           'ba24avg', 'ba24min', 'ba25avg', 'ba25min', 'ba26avg', 'ba27avg', 'ba27max', 'ba27min', 'ba27num',
           'ba28avg', 'ba28max', 'ba28min']
btList = ['btposavg', 'btnegavg']
radList = ['rad']
neighList = ['gcn', 'scn']  # "sgcn"
orderList = ['centParam', 'entroParam', 'entroAvgParam']  # "angParam", "Ixx", "Iyy", "Izz", "degenDeg"
chiList = ['chi5', 'chi6']  # "chi1"
# qList = ['q2avg', 'q4avg', 'q6avg', 'q8avg', 'q10avg', 'q12avg']
# disorderList = ['disord2', 'disord4', 'disord6', 'disord8', 'disord10', 'disord12', 
#                 'disordAvg4', 'disordAvg2', 'disordAvg6', 'disordAvg8', 'disordAvg10', 'disordAvg12']  # All of them do not satisfy the invariance
posList = radList
geomList = blList + ba1List + ba2List + btList + chiList
# steinList = qList  # + disorderList

# Generate combinations of features for game theoretical approach
allFeat = [tuple(posList), tuple(geomList), tuple(neighList), tuple(orderList)]
#combList = []
#for i in range(len(allFeat), 0, -1):
#    combObj = combinations(allFeat, i)
#    for (j, comb) in enumerate(combObj):
#        cList = []
#        for tup in comb: cList += list(tup)
#        combList.append(cList)
combList = list(dfNoLVHCs[0].columns)
#for (i, comb) in enumerate(combList): print(f"Combination {i}: {comb}")
#print(f"Total number of combinations: {len(combList)}")



# Define global variables
cmap = 'jet'
attDict = {"x": {"name": "x", "leg": []}, 
           "y": {"name": "y", "leg": []}, 
           "z": {"name": "z", "leg": []}, 
           "blavg": {"name": "Average Bond Length", "leg": []}, 
           "blmax": {"name": "Maximum Bond Length", "leg": []}, 
           "blmin": {"name": "Minimum Bond Length", "leg": []}, 
           "blnum": {"name": "Number of Bonds", "leg": []}, 
           "bl1avg": {"name": "Average Bond Length11", "leg": []}, 
           "bl1max": {"name": "Maximum Bond Length11", "leg": []}, 
           "bl1min": {"name": "Minimum Bond Length11", "leg": []}, 
           "bl1num": {"name": "Number of Bonds11", "leg": []}, 
           "bl2avg": {"name": "Average Bond Length12", "leg": []}, 
           "bl2max": {"name": "Maximum Bond Length12", "leg": []}, 
           "bl2min": {"name": "Minimum Bond Length12", "leg": []}, 
           "bl2num": {"name": "Number of Bonds12", "leg": []}, 
           "bl3avg": {"name": "Average Bond Length21", "leg": []}, 
           "bl3max": {"name": "Maximum Bond Length21", "leg": []}, 
           "bl3min": {"name": "Minimum Bond Length21", "leg": []}, 
           "bl3num": {"name": "Number of Bond21", "leg": []}, 
           "bl4avg": {"name": "Average Bond Length22", "leg": []}, 
           "bl4max": {"name": "Maximum Bond Length22", "leg": []}, 
           "bl4min": {"name": "Minimum Bond Length22", "leg": []}, 
           "bl4num": {"name": "Number of Bonds22", "leg": []}, 
           "ba1avg": {"name": "Average First Order Bond Angle", "leg": []}, 
           "ba1max": {"name": "Maximum First Order Bond Angle", "leg": []}, 
           "ba1min": {"name": "Minimum First Order Bond Angle", "leg": []}, 
           "ba1num": {"name": "Number of First Order Bond Angles", "leg": []}, 
           "ba11avg": {"name": "Average First Order Bond Angle1", "leg": []}, 
           "ba11max": {"name": "Maximum First Order Bond Angle1", "leg": []}, 
           "ba11min": {"name": "Minimum First Order Bond Angle1", "leg": []}, 
           "ba11num": {"name": "Number of First Order Bond Angles1", "leg": []}, 
           "ba12avg": {"name": "Average First Order Bond Angle2", "leg": []}, 
           "ba12max": {"name": "Maximum First Order Bond Angle2", "leg": []}, 
           "ba12min": {"name": "Minimum First Order Bond Angle2", "leg": []}, 
           "ba12num": {"name": "Number of First Order Bond Angles2", "leg": []}, 
           "ba13avg": {"name": "Average First Order Bond Angle3", "leg": []}, 
           "ba13max": {"name": "Maximum First Order Bond Angle3", "leg": []}, 
           "ba13min": {"name": "Minimum First Order Bond Angle3", "leg": []}, 
           "ba13num": {"name": "Number of First Order Bond Angles3", "leg": []}, 
           "ba14avg": {"name": "Average First Order Bond Angle4", "leg": []}, 
           "ba14max": {"name": "Maximum First Order Bond Angle4", "leg": []}, 
           "ba14min": {"name": "Minimum First Order Bond Angle4", "leg": []}, 
           "ba14num": {"name": "Number of First Order Bond Angles4", "leg": []}, 
           "ba15avg": {"name": "Average First Order Bond Angle5", "leg": []}, 
           "ba15max": {"name": "Maximum First Order Bond Angle5", "leg": []}, 
           "ba15min": {"name": "Minimum First Order Bond Angle5", "leg": []}, 
           "ba15num": {"name": "Number of First Order Bond Angles5", "leg": []}, 
           "ba16avg": {"name": "Average First Order Bond Angle6", "leg": []}, 
           "ba16max": {"name": "Maximum First Order Bond Angle6", "leg": []}, 
           "ba16min": {"name": "Minimum First Order Bond Angle6", "leg": []}, 
           "ba16num": {"name": "Number of First Order Bond Angles6", "leg": []}, 
           "ba17avg": {"name": "Average First Order Bond Angle7", "leg": []}, 
           "ba17max": {"name": "Maximum First Order Bond Angle7", "leg": []}, 
           "ba17min": {"name": "Minimum First Order Bond Angle7", "leg": []}, 
           "ba17num": {"name": "Number of First Order Bond Angles7", "leg": []}, 
           "ba18avg": {"name": "Average First Order Bond Angle8", "leg": []}, 
           "ba18max": {"name": "Maximum First Order Bond Angle8", "leg": []}, 
           "ba18min": {"name": "Minimum First Order Bond Angle8", "leg": []}, 
           "ba18num": {"name": "Number of First Order Bond Angles8", "leg": []}, 
           "ba2avg": {"name": "Average Second Order Bond Angle", "leg": []}, 
           "ba2max": {"name": "Maximum Second Order Bond Angle", "leg": []}, 
           "ba2min": {"name": "Minimum Second Order Bond Angle", "leg": []}, 
           "ba2num": {"name": "Number of Second Order Bond Angles", "leg": []}, 
           "ba21avg": {"name": "Average Second Order Bond Angle1", "leg": []}, 
           "ba21max": {"name": "Maximum Second Order Bond Angle1", "leg": []}, 
           "ba21min": {"name": "Minimum Second Order Bond Angle1", "leg": []}, 
           "ba21num": {"name": "Number of Second Order Bond Angles1", "leg": []}, 
           "ba22avg": {"name": "Average Second Order Bond Angle2", "leg": []}, 
           "ba22max": {"name": "Maximum Second Order Bond Angle2", "leg": []}, 
           "ba22min": {"name": "Minimum Second Order Bond Angle2", "leg": []}, 
           "ba22num": {"name": "Number of Second Order Bond Angles2", "leg": []}, 
           "ba23avg": {"name": "Average Second Order Bond Angle3", "leg": []}, 
           "ba23max": {"name": "Maximum Second Order Bond Angle3", "leg": []}, 
           "ba23min": {"name": "Minimum Second Order Bond Angle3", "leg": []}, 
           "ba23num": {"name": "Number of Second Order Bond Angles3", "leg": []}, 
           "ba24avg": {"name": "Average Second Order Bond Angle4", "leg": []}, 
           "ba24max": {"name": "Maximum Second Order Bond Angle4", "leg": []}, 
           "ba24min": {"name": "Minimum Second Order Bond Angle4", "leg": []}, 
           "ba24num": {"name": "Number of Second Order Bond Angles4", "leg": []}, 
           "ba25avg": {"name": "Average Second Order Bond Angl5e", "leg": []}, 
           "ba25max": {"name": "Maximum Second Order Bond Angle5", "leg": []}, 
           "ba25min": {"name": "Minimum Second Order Bond Angle5", "leg": []}, 
           "ba25num": {"name": "Number of Second Order Bond Angles5", "leg": []}, 
           "ba26avg": {"name": "Average Second Order Bond Angle6", "leg": []}, 
           "ba26max": {"name": "Maximum Second Order Bond Angle6", "leg": []}, 
           "ba26min": {"name": "Minimum Second Order Bond Angle6", "leg": []}, 
           "ba26num": {"name": "Number of Second Order Bond Angles6", "leg": []}, 
           "ba27avg": {"name": "Average Second Order Bond Angle7", "leg": []}, 
           "ba27max": {"name": "Maximum Second Order Bond Angle7", "leg": []}, 
           "ba27min": {"name": "Minimum Second Order Bond Angle7", "leg": []}, 
           "ba27num": {"name": "Number of Second Order Bond Angles7", "leg": []}, 
           "ba28avg": {"name": "Average Second Order Bond Angle8", "leg": []}, 
           "ba28max": {"name": "Maximum Second Order Bond Angle8", "leg": []}, 
           "ba28min": {"name": "Minimum Second Order Bond Angl8e", "leg": []}, 
           "ba28num": {"name": "Number of Second Order Bond Angles8", "leg": []}, 
           "btposavg": {"name": "Average Positive Bond Torsion", "leg": []}, 
           "btposmax": {"name": "Maximum Positive Bond Torsion", "leg": []}, 
           "btposmin": {"name": "Minimum Positive Bond Torsion", "leg": []}, 
           "btposnum": {"name": "Number of Positive Bond Torsions", "leg": []}, 
           "btnegavg": {"name": "Average Negative Bond Torsion", "leg": []}, 
           "btnegmax": {"name": "Maximum Negative Bond Torsion", "leg": []}, 
           "btnegmin": {"name": "Minimum Negative Bond Torsion", "leg": []}, 
           "btnegnum": {"name": "Number of Negative Bond Torsions", "leg": []}, 
           "cn": {"name": "Coordination Number", "leg": []}, 
           "gcn": {"name": "Generalised Coordination Number", "leg": []}, 
           "scn": {"name": "Surface Coordination Number", "leg": []}, 
           "sgcn": {"name": "Surface Generalised Coordination Number", "leg": []}, 
           "rad": {"name": "Radial Distance", "leg": []}, 
           "q6q6": {"name": "Number of Q6Q6 Neighbours", "leg": []}, 
           "angParam": {"name": "Angular Parameter", "leg": []}, 
           "centParam": {"name": "Centrosymmetry Parameter", "leg": []}, 
           "entroParam": {"name": "Entropy Parameter", "leg": []}, 
           "entroAvgParam": {"name": "Average Entropy Parameter", "leg": []},
           "chi1": {"name": "Chi Parameters 1", "leg": []}, 
           "chi2": {"name": "Chi Parameters 2", "leg": []}, 
           "chi3": {"name": "Chi Parameters 3", "leg": []}, 
           "chi4": {"name": "Chi Parameters 4", "leg": []},
           "chi5": {"name": "Chi Parameters 5", "leg": []}, 
           "chi6": {"name": "Chi Parameters 6", "leg": []}, 
           "chi7": {"name": "Chi Parameters 7", "leg": []},
           "chi8": {"name": "Chi Parameters 8", "leg": []}, 
           "chi9": {"name": "Chi Parameters 9", "leg": []}, 
           "q2": {"name": "Q2 Values", "leg": []}, 
           "q4": {"name": "Q4 Values", "leg": []}, 
           "q6": {"name": "Q6 Values", "leg": []}, 
           "q8": {"name": "Q8 Values", "leg": []},
           "q10": {"name": "Q10 Values", "leg": []}, 
           "q12": {"name": "Q12 Values", "leg": []}, 
           "q2avg": {"name": "Averaged Q2 Values", "leg": []}, 
           "q4avg": {"name": "Averaged Q4 Values", "leg": []}, 
           "q6avg": {"name": "Averaged Q6 Values", "leg": []}, 
           "q8avg": {"name": "Averaged Q8 Values", "leg": []},
           "q10avg": {"name": "Averaged Q10 Values", "leg": []}, 
           "q12avg": {"name": "Averaged Q12 Values", "leg": []}, 
           "disord2": {"name": "Q2 Disorder Parameter", "leg": []}, 
           "disord4": {"name": "Q4 Disorder Parameter", "leg": []}, 
           "disord6": {"name": "Q6 Disorder Parameter", "leg": []}, 
           "disord8": {"name": "Q8 Disorder Parameter", "leg": []}, 
           "disord10": {"name": "Q10 Disorder Parameter", "leg": []}, 
           "disord12": {"name": "Q12 Disorder Parameter", "leg": []}, 
           "disordAvg2": {"name": "Q2 Average Disorder Parameter", "leg": []}, 
           "disordAvg4": {"name": "Q4 Average Disorder Parameter", "leg": []}, 
           "disordAvg6": {"name": "Q6 Average Disorder Parameter", "leg": []}, 
           "disordAvg8": {"name": "Q8 Average Disorder Parameter", "leg": []}, 
           "disordAvg10": {"name": "Q10 Average Disorder Parameter", "leg": []}, 
           "disordAvg12": {"name": "Q12 Average Disorder Parameter", "leg": []}, 
           "Ixx": {"name": "Moment of Inertia X Direction", "leg": []}, 
           "Iyy": {"name": "Moment of Inertia Y Direction", "leg": []}, 
           "Izz": {"name": "Moment of Inertia Z Direction", "leg": []}, 
           "degenDeg": {"name": "Degeneracy Degree", "leg": []}, 
           "surf": {"name": "Surface", "leg": []}, 
           "order": {"name": "Order", "leg": []}, 
           "ILSclustAll": {"name": "ILS All Cluster Label", "leg": []}, 
           "ILSclustSurf": {"name": "ILS Surface Cluster Label", "leg": []}}



# ## PCA

def pca(X,n):
    pca = PCA(n_components = n)
    X = pca.fit_transform(X)



# ## ILS Clustering


def calcRhoDeltaGamma(X, verbose=False):
    """
    Returns rho (density), delta (min distance from points with higher rho), gamma (scores) for 
    Clustering by Fast Search and Finding of Density Peaks (CFSDP) (adapted from Thea Hsu's codes).
    
    Inputs:
        X : DataFrame with each column being a feature and each row being an atom instance
        verbose : Boolean determining whether print statements are activated
    Outputs:
        density : NumPy matrix representing density of N points (rho)
        neighDensPoints : NumPy matrix representing computing the minimum distance between the point and any other with high density (delta)
        scores : NumPy matrix representing product of rho and delta (gamma)
    """
    
    # Choose cutoff distance for computation of rho and delta non-parametrically
    distanceMatrix = pairwise_distances(X)
    dc, dcValueList, field = cfs.choose_dc(distanceMatrix)
    if verbose: print(f"    Cutoff distance to compute ρ and δ: {dc:.3f}")  
    
    if verbose: print("    Computing ρ, δ, and γ...")
    density = cfs.continuous_density(distanceMatrix, dc)  # rho
    neighDensPoints = cfs.delta_function(distanceMatrix, density)  # delta
    scores = cfs.choosing_centernumber(density, neighDensPoints)  # gamma
    
    return density, neighDensPoints, scores



def initILS(X, caseID, combID, initL, attList=['order', 'surf', 'cn'], allOrSurf='Surface', figSize=(16, 5), verbose=False):
    """
    Initialise a point at 'initL', run ILS over atoms specified by 'allOrSurf', and colour the plot by 
    attributes listed in 'attList'.
    
    Inputs:
        X : DataFrame with each column being a feature and each row being an atom instance
        caseID : Integer indicating index number of test case
        combID : Integer indicating index number of feature combination
        initL : (List of) initial label for ILS (correspond to index of X instead of row number)
        attList : List of features present in the targeted DataFrame to colour the ILS plots with
        allOrSurf : String indicating the portion of atoms of interest {'All', 'Surface'}
        verbose : Boolean determining whether print statements are activated
    Outputs:
        newL : Series of the labels of each sample; ordered by the labelling order
        orderedL : DataFrame (ordered by the labelling order) containing 
                    minR : Float representing high-dimensional distance from the previous nearest labelled neighbour (label source)
                    IDclosestLabel : Float representing the cluster label of the previous nearest labelled neighbour (label source)
                    order : Integer representing the labelling order of each sample
    """
    if verbose: print(f"  Running ILS on {allOrSurf} atoms, initial labelled index: {initL}")
    X = X.copy()
    XnoScale = dfNoLVHCs[caseID][dfNoLVHCs[caseID]["surf"] == 1] if allOrSurf == "Surface" else dfNoLVHCs[caseID]
    attLists = [tuple(attList[i:i+3]) for i in range(0, len(attList), 3)]

    # Initialise a label
    X['label'] = 0
    X.loc[initL, 'label'] = 1
    X.index.name = 'ID'
    newL, orderedL = ILS(X, 'label')
    orderedL['order'] = range(len(orderedL))
    
    for (i, attList) in enumerate(attLists):
        fig, axs = plt.subplots(1, 3, figsize=figSize);
        for (j, ax) in enumerate(axs):
            colorList = range(len(orderedL)) if attList[j] == 'order' else XnoScale[attList[j]][list(orderedL.index)]
            ax.plot(orderedL['order'], orderedL['minR'], color='k', marker=None, linestyle='solid', linewidth=0.3, zorder=1);
            scatter = ax.scatter(orderedL["order"], orderedL["minR"], c=colorList, cmap=cmap, s=24, zorder=2);
            # sns.scatterplot(data=orderedL, x='order', y='minR', hue='order', palette='flare', legend=False);
            if attList[j] == 'surf':
                ax.legend(handles=scatter.legend_elements()[0], labels=('Bulk', 'Surf'));  # title='Surface'
            elif attList[j] == 'cn':
                ax.legend(handles=scatter.legend_elements()[0], labels=sorted(XnoScale["cn"].unique()));  # title="Coordination Number"
            else:
                plt.colorbar(scatter, orientation='vertical', fraction=0.05, shrink=1.0, aspect=20, pad=0.02, ax=ax, anchor=(1, 0.5))
            ax.set_title(f"Coloured by {attDict[attList[j]]['name']}");
        plt.suptitle(f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms ILS Distance Plots");
        fig.add_subplot(111, frameon=False)
        plt.grid(False);
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Order Labelled, $i$");
        plt.ylabel(r"Minimum Distance, $R_{min}(i)$");
        plt.tight_layout();
        plt.savefig(fname=f"figures/initILS/initILS80_{npName}_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
        plt.show();
    if verbose: print(f"    Saved the rMin plots for {allOrSurf} atoms!")
    
    return newL, orderedL


def findPeaks(minR, halfWinSize, peakFunc='S2', sigConst=1.2, diffThresh=0.01, verbose=False):
    """
    Locate maxima in a series of values.
    
    Inputs:
        minR : Series of minimum distance values as obtained from ILS clustering output (np array of length N)
        halfWinSize (k) : Integer indicating half of the range of values to be averaged over
        peakFunc : String indicating choice of peak function to use for peak detection
        sigConst (h): Float of significance constant, typically 1.0<=h<=3.0, can go negative (if negative, 
                      magnitude proportional to number of peaks captured)
        diffThresh: Float of threshold that specifies if 2 peaks are different enough in the case of ordered nanoparticles
        verbose : Boolean determining whether print statements are activated
    Outputs: 
        maxima : 1D NumPy array consisting of detected peaks (in terms of order and not index, mainly due to practical purposes -- 
                 easier validation via visualisation)
    """
    if verbose: print(f"    Finding peaks using peak function {peakFunc}")
    
    peakFuncVals = np.zeros(len(minR))
    for (i, R) in enumerate(minR):
        if i == 0 or i == len(minR)-1: continue
        
        # Extract neighbouring points from both sides
        leftStart = 0 if i-halfWinSize < 0 else i-halfWinSize
        rightEnd = None if i+halfWinSize > len(minR) else i+halfWinSize
        leftWindow, rightWindow = minR[leftStart:i], minR[i+1:rightEnd]
        leftDiff, rightDiff = R - leftWindow, R - rightWindow 
        
        # Compute peak functions
        if peakFunc == 'S1':
            peakFuncVal = (leftDiff.max()+rightDiff.max()) / 2
        elif peakFunc == 'S2':
            peakFuncVal = (leftDiff.sum()/halfWinSize+rightDiff.sum()/halfWinSize) / 2
        elif peakFunc == 'S3':
            peakFuncVal = (R-leftWindow.sum()/halfWinSize+R-rightWindow.sum()/halfWinSize) / 2
        else:
            raise AssertionError("    Peak function specified is unknown!")
        peakFuncVals[i] = peakFuncVal
    
    # Peak candidates have positive peak function values, compute mean & std of candidates' values to determine their global significance
    posPeakFuncVals, nonPosPeakFuncVals = peakFuncVals[peakFuncVals > 0], peakFuncVals[peakFuncVals <= 0]
    posPeakFuncMean, posPeakFuncStd = posPeakFuncVals.mean(), posPeakFuncVals.std()
    nonPosPeakFuncMean, nonPosPeakFuncStd = nonPosPeakFuncVals.mean(), nonPosPeakFuncVals.std()
    
    # Handle the case when a peak dominates (especially when nanoparticle is ordered)
    handleOrdered = None
    if nonPosPeakFuncMean > NON_POS_PEAK_FUNC_MEAN_THRESH and nonPosPeakFuncStd < NON_POS_PEAK_FUNC_STD_THRESH:  # values hard-coded around S2 based on empirical experience ***
        handleOrdered = 2  # Hard-coded to be option 2 for now ***
        print(f"    Dominating peak(s) present, handling using option {handleOrdered}")
        # Option 1: Adjust global significance of peaks that are being dominated
        if handleOrdered == 1:
            posPeakFuncVals = np.delete(posPeakFuncVals, posPeakFuncVals.argmax())
            posPeakFuncMean, posPeakFuncStd = posPeakFuncVals.mean(), posPeakFuncVals.std()
    # Option 2: Turn the significance constant into a (visibility) threshold
    if handleOrdered == 2:
        peakFuncValThresh = posPeakFuncVals.max() * sigConst/SIG_CONST_SCALER   # Could potentially find a better function for this transformation ***
    else:
        peakFuncValThresh = sigConst*posPeakFuncStd + posPeakFuncMean
    
    maxima, prevPeakPos = [], -halfWinSize
    for (peakPos, peakFuncVal) in enumerate(peakFuncVals):
        if peakFuncVal > peakFuncValThresh:
            if verbose: print(f"    Found qualified peak candidate: {peakPos}, Peak function value: {peakFuncVal:.3f}")
            
            # Handle first peak case to avoid indexing errors in later stages
            if len(maxima) == 0:
                maxima.append(peakPos)
                prevPeakPos = peakPos
                continue
            
            # Handle peaks among similar peaks (in ordered nanoparticles)
            if handleOrdered:
                # if abs(peakFuncVal - peakFuncVals[maxima[-1]]) < diffThresh:
                startIdx = 0 if peakPos-halfWinSize < 0 else peakPos-halfWinSize
                endIdx = None if peakPos+halfWinSize > len(peakFuncVals) else peakPos+halfWinSize
                window = peakFuncVals[startIdx:endIdx]
                similarPeaks = np.where(np.logical_and(window >= peakFuncVal-diffThresh, window <= peakFuncVal+diffThresh))[0]
                if len(similarPeaks) > 1:
                    if verbose: print("      But too similar to surrounding peaks, current peak removed.")
                    continue
                    
            # Choose more significant peak if peak found within window range of previous peak
            if peakPos < prevPeakPos + halfWinSize:
                if verbose: print("      But too close to previous peak, comparing the peaks...")
                if peakFuncVal > peakFuncVals[maxima[-1]]:
                    if verbose: print("      Current peak is greater, previous peak removed.")
                    maxima.pop()
                else:
                    if verbose: print("      Previous peak is greater, current peak removed.")
                    continue
            maxima.append(peakPos)
            prevPeakPos = peakPos
            
    # Optional condition to remove last peak if the cluster is too small
    # if maxima[-1] + halfWinSize > len(minR): maxima.pop()  # Doesn't need to be 'halfWinSize' here
        
    return np.array(maxima)




def locateDensestIdx(orderedL, halfWinSize, sigConst=1.2, verbose=False):
    """
    Locate densest point in a series of values based on the corresponding gamma scores.
    
    Inputs:
        orderedL : DataFrame (ordered by the labelling order) containing 
                    minR : Float representing high-dimensional distance from the previous nearest labelled neighbour (label source)
                    IDclosestLabel : Float representing the cluster label of the previous nearest labelled neighbour (label source)
                    order : Integer representing the labelling order of each sample
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
        halfWinSize : Integer indicating half of the range of values to be averaged over
        sigConst (h): Float of significance constant, typically 1.0<=h<=3.0, can go negative (if negative, 
                      magnitude proportional to number of peaks captured)
        verbose : Boolean determining whether print statements are activated
    Outputs:
        endPointsOrders : List of maxima in the series + the last point of the series (in terms of order instead of index)
        clustCentsOrders : List of minima in the series (in terms of order instead of index)
    """
    index = np.arange(len(orderedL))
    # Find all peaks, then identify the order and indices of atoms in between the peaks (clustering up)
    peaksOrders = findPeaks(minR=np.array(orderedL['minR']), halfWinSize=halfWinSize, peakFunc=PEAK_FUNC, sigConst=sigConst, diffThresh=DIFF_THRESH, verbose=verbose)
    gammaBtwPeaksAllClusts, ordersBtwPeaksAllClusts = np.split(np.array(orderedL['gamma']), peaksOrders), np.split(index, peaksOrders)
    
    # Identify densest points in each cluster based on gamma scores, in terms of index in numpy array (essentially order), not index in orderedLsurf
    subClustCentIdxs = [np.argmax(gammaBtwPeaks) for gammaBtwPeaks in gammaBtwPeaksAllClusts]
    
    # Return the order (not index, for easier validation via visualisation) of the endpoints and centres of each cluster
    endPointsOrders = [ordersBtwPeaks[-1] for ordersBtwPeaks in ordersBtwPeaksAllClusts]
    clustCentsOrders = [ordersBtwPeaksAllClusts[clustIdx][subClustCent] for (clustIdx, subClustCent) in enumerate(subClustCentIdxs)]
    if verbose: print(f"    End points (in terms of order): {endPointsOrders}\n    Cluster centres (in terms of order): {clustCentsOrders}")
    return endPointsOrders, clustCentsOrders



def findClusts(X, caseID, combID, orderedL, allOrSurf='Surface', winSize=50, sigConst=1.2, minClustSize=10, verbose=False):
    """
    Find the cluster centres based on the peaks in orderedL, label them as initial point, and run ILS to cluster similar atoms together.
    
    Inputs:
        X : DataFrame with each column being a feature and each row being an atom instance
        caseID : Integer indicating index number of test case
        combID : Integer indicating index number of feature combination
        orderedL : DataFrame (ordered by the labelling order) containing 
                    minR : Float representing high-dimensional distance from the previous nearest labelled neighbour (label source)
                    IDclosestLabel : Float representing the cluster label of the previous nearest labelled neighbour (label source)
                    order : Integer representing the labelling order of each sample
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
        allOrSurf : String indicating the portion of atoms of interest {"All", "Surface"}
        winSize : Integer indicating window range to average for peak detection
        minClustSize : Integer indicating minimum size of a cluster
        verbose : Boolean determining whether print statements are activated
    Outputs:
        ILSclust : Series of ILS cluster labels
        ILSorder : DataFrame (ordered by the labelling order) containing
                    minR : Float representing high-dimensional distance from the previous nearest labelled neighbour (label source)
                    IDclosestLabel : Float representing the cluster label of the previous nearest labelled neighbour (label source)
    """
    X = X.copy()
    X.index.name = 'ID'
    
    if verbose: print(f"  Identifying indices of cluster centres of {allOrSurf} atoms...")
    endPointsOrders, clustCentsOrders = locateDensestIdx(orderedL=orderedL, halfWinSize=winSize//2, sigConst=sigConst, verbose=verbose)
    allCentIdxs = [orderedL.iloc[clustCent, :].name for clustCent in clustCentsOrders]  # centres returned in terms of order instead of index
    
    centIdxs = deepcopy(allCentIdxs)
    removedLabels = []
    while len(centIdxs) > 0:
        if verbose: print(f"  (Re)labelling the cluster centres of {allOrSurf} atoms...")
        X['label'] = 0
        for (i, centIdx) in enumerate(centIdxs):
            X.loc[centIdx, 'label'] = i + 1

        if verbose: print(f"  Applying clustering on {allOrSurf} atoms using ILS...")
        ILSclust, ILSorder = ILS(df=X, labelColumn='label', iterative=True)  # If iterative=False only initial labels are used during spreading
        clusterLabelCounts = ILSclust.value_counts(ascending=False).to_dict()
        labelsToRemove = []
        for clusterLabel in ILSclust.unique():
            if verbose: print(f"    Cluster {clusterLabel}: Counts = {clusterLabelCounts[clusterLabel]}")
            if clusterLabelCounts[clusterLabel] < minClustSize: labelsToRemove.append(int(clusterLabel))
            
        # Remove cluster centres for clusters that are too small before relabelling them and rerunning ILS
        if len(labelsToRemove) > 0: 
            if verbose: print(f"  Removing cluster(s) < {minClustSize} in size!")
            centIdxs = [centIdx for (i, centIdx) in enumerate(allCentIdxs) if i+1 not in labelsToRemove]
            removedLabels = deepcopy(labelsToRemove)
        else: break
    
    # Mark the cluster centres and endpoints
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(len(orderedL)), orderedL['minR'], color='k', marker='o', markersize=2, linestyle='solid', linewidth=1, zorder=-1);
    for endPoint in endPointsOrders: plt.axvline(endPoint, c='r');
    for (i, centOrder) in enumerate(clustCentsOrders): 
        if i+1 not in removedLabels: plt.plot(centOrder, orderedL['minR'].iloc[centOrder], color='b', marker='*', markersize=15, zorder=0);
        else: plt.plot(centOrder, orderedL['minR'].iloc[centOrder], color='g', marker='*', markersize=15, zorder=0);
    plt.xlabel("Order Labelled, $i$");
    plt.ylabel(r"Minimum Distance, $R_{min}(i)$");
    plt.title(f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms ILS Distance Plots Cluster Centres and Peaks");
    plt.savefig(fname=f"figures/centPeak/centPeak80_{npName}_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
    if verbose: print(f"    Saved the centres & peaks plot for {allOrSurf} atoms!")

    # Colour original ILS distance plot by ILS clusters
    fig, axes = plt.subplots(1, 2, figsize=(10, 5));
    axes[0].plot(range(len(orderedL)), orderedL['minR'], color='k', marker=None, linestyle='solid', linewidth=0.3, zorder=1);
    axes[0].scatter(range(len(orderedL)), orderedL['minR'], c=list(ILSclust.reindex((list(orderedL.index)))), cmap=cmap, s=24, zorder=2);
    axes[0].set_title("Densest Point as Initial Label");
    axes[1].plot(range(len(ILSorder)), ILSorder['minR'], color='k', marker=None, linestyle='solid', linewidth=0.3, zorder=1);
    axes[1].scatter(range(len(ILSorder)), ILSorder['minR'], c=list(ILSclust.drop(centIdxs)), cmap=cmap, s=24, zorder=2);
    axes[1].set_title("Cluster Minima as Initial Labels");
    plt.suptitle(f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms ILS Distance Plots Coloured by ILS Clusters")
    fig.add_subplot(111, frameon=False);
    plt.grid(False);
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False);
    plt.xlabel("Order Labelled, $i$");
    plt.ylabel(r"Minimum Distance, $R_{min}(i)$");
    plt.tight_layout();
    plt.savefig(fname=f"figures/ILSclust/ILSclust80_{npName}_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
    if verbose: print(f"    Saved the rMin plots coloured by cluster labels for {allOrSurf} atoms!")
    
    return ILSclust, ILSorder





def subClustCheck(X, caseID, combID, orderedL, ILSclust, allOrSurf='Surface', verbose=False, clustMethod='Combo'):
    """
    Run ILS on each apparent ILS cluster to check for subclusters within it.
    Relabel all samples by initialising at each cluster minimum point.
    
    Inputs:
        X : DataFrame with each column being a feature and each row being an atom instance
        caseID : Integer indicating index number of test case
        combID : Integer indicating index number of feature combination
        orderedL : DataFrame (ordered by the labelling order) containing 
                    minR : Float representing high-dimensional distance from the previous nearest labelled neighbour (label source)
                    IDclosestLabel : Float representing the cluster label of the previous nearest labelled neighbour (label source)
                    order : Integer representing the labelling order of each sample
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
        ILSclust : DataFrame (sorted by index) containing
                    cluster: Float representing ILS cluster labels
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                    gcn : Float representing GCN values of each sample atom
        allOrSurf : String indicating the portion of atoms of interest {"All", "Surface"}
        verbose : Boolean determining whether print statements are activated
    Outputs: 
        None
    """
    if verbose: print(f"  Checking for existence of subclusters within each cluster of {allOrSurf} atoms...")
    
    # Run ILS for each cluster
    clustLabelsSorted = sorted(ILSclust['cluster'].unique())
    numClusts = len(clustLabelsSorted)
    fig, axes = plt.subplots(1, numClusts, figsize=(4*numClusts, 5));
    for (i, clusterLabel) in enumerate(clustLabelsSorted):
        clustMemIdxs = list(ILSclust[ILSclust['cluster'] == clusterLabel].index)
        Xsub = X.loc[clustMemIdxs, :].copy()
        Xsub['label'] = 0
        initL = ILSclust['gamma'][clustMemIdxs].idxmax()
        Xsub.loc[initL, 'label'] = clusterLabel
        Xsub.index.name = 'ID'
        newLsub, orderedLsub = ILS(df=Xsub, labelColumn='label', iterative=True)
        if numClusts == 1:
            axes.plot(range(len(orderedLsub)), orderedLsub['minR']);
            axes.set_ylim(top=orderedL['minR'].max());
            axes.set_title(f"Cluster {clusterLabel:1.0f}");
        else:
            axes[i].plot(range(len(orderedLsub)), orderedLsub['minR']);
            axes[i].set_ylim(top=orderedL['minR'].max());
            axes[i].set_title(f"Cluster {clusterLabel:1.0f}");
    plt.suptitle(f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms ILS Distance Plots of Each Cluster");
    fig.add_subplot(111, frameon=False);
    plt.grid(False);
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False);
    plt.xlabel("Order Labelled, $i$");
    plt.ylabel(r"Minimum Distance, $R_{min}(i)$");
    plt.tight_layout();
    plt.savefig(fname=f"figures/subClust/subClust{clustMethod}80_{npName}_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
    plt.show();
    if verbose: print(f"    Saved the individual cluster rMin plots for {allOrSurf} atoms!")




def plotGCN(caseID, combID, ILSclust, allOrSurf='Surface', verbose=False, clustMethod='Combo'):
    """
    Plot and save GCN histogram and kernel density estimated distribution for each cluster.
    Methods for choosing optimal bin number or bin width obtained from https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    
    Inputs:
        caseID : Integer indicating index number of test case
        combID : Integer indicating index number of feature combination
        ILSclust : DataFrame (sorted by index) containing
                    cluster: Float representing ILS cluster labels
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                    gcn : Float representing GCN values of each sample atom
        allOrSurf : String indicating the portion of atoms of interest {"All", "Surface"}
        verbose : Boolean determining whether print statements are activated
    Outputs: 
        None
    """
    clustLabelsSorted = sorted(ILSclust['cluster'].unique())
    numClusts = len(clustLabelsSorted)
    
    if verbose: print(f"  Plotting histograms of GCN values of {allOrSurf} atoms clusters...")
    histBinChoices = ['sqrt', 'sturge', 'rice', 'doane', 'scott', 'freedia']
    histBinChoice = histBinChoices[HIST_BIN_CHOICE]  # Current choices: 3 for non-normal data (ordered), 4 for normal data (disordered)
    if verbose: print(f"    Method to choose optimal bin number: {histBinChoice}")
    
    singleGCNclusts = []
    fig, axes = plt.subplots(1, numClusts, figsize=(4*numClusts, 5));
    for (i, clusterLabel) in enumerate(clustLabelsSorted):
        Xsub = dfAlls[caseID].loc[list(ILSclust[ILSclust['cluster'] == clusterLabel].index), :]
        ax = axes[i] if numClusts != 1 else axes
        if len(Xsub['gcn'].unique()) == 1:  
            singleGCNclusts.append((i, Xsub['gcn'].unique()[0]))
        #     # Avoid zero variance, otherwise kernel density estimation in sns.histplot() will get singular matrix error (uncomment if kde=True)
        #     Xsub["gcn"].iloc[0] += 0.01  # Change GCN value of first sample in the cluster 
        #     ILSclust.loc[Xsub.iloc[0].name]["gcn"] += 0.01  # Change the corresponding GCN value in ILSclust, not an ideal approach
        #     Xsub["gcn"].iloc[1] -= 0.01 
        #     ILSclust.loc[Xsub.iloc[1].name]["gcn"] -= 0.01
            
        # Adjust parameters (either number or width of bin) for sns.histplot()
        binNum, binWidth = 'auto', None
        if histBinChoice == 'sqrt':  # Square-root choice
            binNum = math.ceil(math.sqrt(len(Xsub)))
        elif histBinChoice == 'sturge':  # Sturges' formula, assume approx normal distribution, poor for n<30 and large n, optimal for n~100
            binNum = math.ceil(math.log2(len(Xsub))) + 1
        elif histBinChoice == 'rice':  # Rice Rule, simple alternative to Sturge's
            binNum = 2 * math.ceil(np.cbrt(len(Xsub)))
        elif histBinChoice == 'doane':  # Doane's formula, modified Sturge to improve performance on non-normal data
            # print("here",Xsub['gcn'].isnull().values.any())
            binNum = 1 + math.ceil(math.log2(len(Xsub)) + 
                                   math.log2(1+abs(skew(Xsub['gcn'])) / math.sqrt(6*(len(Xsub)-2)/(len(Xsub)+1)/(len(Xsub)+3))))
        elif histBinChoice == 'scott':  # Scott's normal reference rule, optimal for random samples of normal distribution
            binWidth = 3.49 * Xsub['gcn'].std() / np.cbrt(len(Xsub))
        elif histBinChoice == 'freedia':  # Freedman–Diaconis' choice, didn't work for data containing only 1 unique value
            binWidth = 2 * iqr(x=Xsub['gcn'], rng=(25, 75), interpolation='midpoint') / np.cbrt(len(Xsub))
        if binWidth is not None and binWidth < sys.float_info.min: binWidth = 0.02
        if verbose: print(f"    Cluster: {clusterLabel}, Bin number: {binNum}, Bin width: {binWidth}")
        
        # Setting kde=True on might cause plots with very low variance to not show up
        sns.histplot(data=Xsub, x='gcn', ax=ax, bins=binNum, binwidth=binWidth, kde=False, kde_kws={'bw_adjust': 2})  
        ax.set(xlabel='', ylabel='', title=f"Cluster {clusterLabel:1.0f}");
        # avgShiftedHist = ash(ILSclust["gcn"])
        # ax.plot(avgShiftedHist.ash_mesh, avgShiftedHist.ash_den)  # Density estimation via average shifted histogram
    plt.suptitle(f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms GCN Histogram of Each Cluster");
    fig.add_subplot(111, frameon=False);
    plt.grid(False);
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False);
    plt.xlabel("Generalised Coordination Number");  # Switch off if sns.histplot is used.
    plt.ylabel("Count");
    plt.tight_layout();
    plt.savefig(fname=f"figures/gcnHist/gcnHist{clustMethod}80_{npName}_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
    plt.show();
    if verbose: print(f"    Saved the individual cluster GCN histograms for {allOrSurf} atoms!")
    
    # Single plot containing all estimated densities of GCN for each cluster
    # fig, ax = plt.subplots(figsize=(7, 4.5))
    # kdePlot = sns.kdeplot(data=ILSclust, x='gcn', hue='cluster', common_norm=True, bw_adjust=2, palette='deep');  # setting common_norm to True means sum(area) = 1 
    # sns.move_legend(kdePlot, 'upper left', bbox_to_anchor=(1.02, 1), title='Cluster');
    # for (clusterNum, gcnVal) in singleGCNclusts: ax.axvline(x=gcnVal, c=sns.color_palette()[clusterNum]);
    # ax.set(xlabel="Generalised Coordination Number", 
    #        title=f"{testCases[caseID]} Combination {combID} {allOrSurf} Atoms GCN PDF of Each Cluster' ' ");
    # plt.savefig(fname=f"figures/gcnKDE/gcnKDE_{testCases[caseID]}_C{combID}_{allOrSurf}.png");
    # plt.show();
    # if verbose: print(f"    Saved the estimated GCN PDFs for each individual cluster for {allOrSurf} atoms!")


# ## Kmeans++ Clustering (Initialisation optimisation)


def initialise_parameters(m, n, X):
    C = np.zeros((m,n))
    
    #apply kmeans++ method for initialisation
    '''compared to other initialisation methods, kmeans++ behaves better in accuracy and speed, 
    and also it only randomize the first centroid, avoiding the case that two randomly-generated 
    centroids are quite close to each other, thus leading to overlapping later because it picks the 
    farthest point as next centroid'''
    
    C_list=[] # intialise a list to store centroids
    #first randomise the first centroid
    nn=X.shape[0]
    chosenNum = np.random.randint(nn)
    C_list.append(X[chosenNum, :])  
    
    # obtain remaining m - 1 centroids 
    for centre_index in range(0,m - 1): 
          
        # initialise a list to store distances between a data point and its nearest centroid 
        distance = [] 
        for data_point in X: 
            if not is_Datapoint_already_a_centroid(data_point,C_list): # remove the already selected centroid (avoid two duplicate centroids)
                closest_dist =  Euclidean_distance(data_point,C_list[0]) # initialise the closest distance to be the distance with first centroid
              
                # calculate distance between point and each previously-selected centroid then save the closest distance 
                for centre in C_list: 
                    temp_dist = Euclidean_distance(data_point,centre) 
                    if temp_dist < closest_dist:
                        closest_dist = temp_dist # update the closest distance
                distance.append(closest_dist) 
              
        # select data_point with maximum distance as our next centre as the probability for selecting is proportional to the distance
        next_centre = X[np.argmax(np.array(distance))] # maximum distance will induce largest probability
        C_list.append(next_centre) 
    C = np.array(C_list)
    return C


def Euclidean_distance(X,Y):
    summation = 0;
    for i in range(0,len(X)):
        summation+=(X[i] - Y[i])**2
    dist = math.sqrt(summation)
    return dist


def is_Datapoint_already_a_centroid(x,y):
    # help avoid duplicate centroids by judging whether a point has been selected as a centroid
    for centre in y:
        centre_array = np.array(centre)
        if (np.array_equal(x,centre_array)):
            return True
    return False


def E_step(C, X):
    
    # E_step is to assign data points to clusters based on the current centroids 
    L_list = [] # initialise a list to store the lables of each data point
    for data_point in X:
        closest_distance = Euclidean_distance(data_point, C[0])  # initialise the closest distance to be the distance with first centroid 
        closest_centroid = C[0] # initialise the label of this data point to be the first centroid
        for centroid in C:
            distance_to_centroid = Euclidean_distance(data_point, centroid)
            if distance_to_centroid < closest_distance:
                closest_distance = distance_to_centroid #update the distance from this data point to its according centroid
                closest_centroid = centroid # update the label
        L_list.append(closest_centroid)  
    L=np.array(L_list)
    return L


def M_step(C, X, L):
    # choosing centroids (center of a cluster) 
    # based on current assignment of data points to clusters.
    emptyClusterCentroids = []
    for (i, centroid) in enumerate(C):
        datapoint_list = [] # initialise a list to later store the data points assigned to this centroid
        
        # determine which data points belong to this centroid based on their label generated from E_step
        for label_index in range(0,len(L)): 
            if np.array_equal(centroid,L[label_index]): # judge whether a data point with specific label belongs to this centroid
                datapoint_list.append(X[label_index])
        
        # recalculate means (new-centroids) for observations assigned to each cluster.
        for dimension in range(0,len(X[0])):
            summation_on_each_dimension = 0
            for data_point in datapoint_list:
                summation_on_each_dimension+=data_point[dimension]
            if len(datapoint_list) < MIN_CLUST_SIZE: emptyClusterCentroids.append(i)
            else: centroid[dimension] = summation_on_each_dimension/len(datapoint_list) # coordinate of updated centroid on a dimension is average value of those of data points assigned to it
    if len(emptyClusterCentroids) > 0: C = np.delete(C, emptyClusterCentroids, 0)
    return C 



def kmeans(X, m, i):
    '''
    Inputs:
    X: Dataset 
    m: Designated cluster number
    i: iteration number
    Output: return the dataframe with cluster index and the centroid list
    '''
    OriginalFeaList = X.columns
    XNp = np.array(X)
    #L = np.zeros(XNp.shape)
    #C = np.zeros((m, XNp.shape[1]))
    n = len(XNp[0]) # the dimension of the data points
    C = initialise_parameters(m, n, XNp) # initialise the centroids for the clustering process
    for iteration in range(0,i): # iterates between E and M steps i times
        L = E_step(C, XNp) #labels (centroid values) assigned to each sample
        C = M_step(C, XNp, L) #centers of the clusters
    L = E_step(C, XNp) #labels (centroid values) assigned to each sample
    ll = []
    for l in L:
        label = np.where((C == l).all(axis=1))[0][0]
        ll.append(label)
    kmCluster = pd.DataFrame(XNp)
    # convert back to pandas dataframe with header
    FeaList = np.reshape(OriginalFeaList,(-1,len(OriginalFeaList)))
    XDF = pd.DataFrame(np.concatenate([FeaList, kmCluster],axis=0))
    header_row = XDF.iloc[0]
    XClusteredDF = pd.DataFrame(XDF.values[1:], columns=header_row)
    
    XClusteredDF['cluster'] = ll
    return XClusteredDF, C



def potFromGCN(gcns, gradList, intercList):
    """
    Compute normalised potential values (which is proportional to the usefulness of the atoms) corresponding to GCN values 
    based on a set of linear functions.
    
    Inputs:
        gcns : List of Floats representing gcn values 
        gradList : List of Floats representing gradients of the set of linear functions
        intercList : List of Floats representing y-intercepts of the set of linear functions
    Outputs:
        normPotVals : 1D NumPy array of Floats representing normalised potential values corresponding to input GCN values
    """
    
    potVals = []
    for gcn in gcns:
        potVal = min([m*gcn + c for m, c in zip(gradList, intercList)])  # The distribution is consisted of the min of the linear lines
        potVals.append(potVal)
    potVals = np.array(potVals)
    
    # Normalise the height (range from 0 to 1) and make sure all values are positive
    normPotVals = (potVals - potVals.min()) / (potVals.max() - potVals.min())
    
    # Normalise potential values based on the area under the distribution curve
    # Area computed using Simpson's composite integration, could consider Romberg's method for more accurate result
    totalArea = simps(y=normPotVals, x=GCN_XRANGE, axis=-1, even='avg')
    normPotVals /= totalArea
    
    return normPotVals



def calcSelSpcSen(p, q):
    """
    Compute selectivity, specificity, and sensitivity of GCN distribution p towards reference reaction GCN-activity distribution q:
        Selectivity is related to the difference in the GCN values corresponding to the peaks of both distributions.
        Specificity is related to the full width at half maximum (FWHM) of the overlapping distributions.
        Sensitivity is related to the maxima of the overlapping distribution.
    1 represent maximum and 0 represent minimum for each value.
    ***DEFINITIONS OF SEL, SPC, SEN YET TO BE REFINED***
    
    Inputs:
        p : NumPy array representing probability distributions (GCN distribution of a given cluster)
        q : NumPy array representing probability distributions (reference reaction GCN-activity distributio)
    Outputs:
        sel : Float representing the computed selectivity
        spc : Float representing the computed specificity
        sen : Float representing the computed sensitivity
    """
    # Compute selectivity, max when peak on top of each other, min when each peak is at the extreme boundaries of GCN range considered
    # Assumptions: The peaks of a distribution represent the bulk of it, essentially asking how close is the bulk of p to the bulk of q
    sel = 1 - abs(GCN_XRANGE[p.argmax()] - GCN_XRANGE[q.argmax()]) / (GCN_HIGH_BOUND - GCN_LOW_BOUND)
    
    # Obtain the overlapping area between distributions p and q
    overlapPotVals, overlapGCNs = [], []
    for (i, gcn) in enumerate(GCN_XRANGE):
        if p[i] != 0 and q[i] != 0:
            overlapPotVals.append(min(p[i], q[i]))
            overlapGCNs.append(gcn)
    overlapPotVals = np.array(overlapPotVals)
    
    # Compute specificity, max when (certain fraction of) p is entirely within (certain fraction of) q, min when there is no overlap between them
    # There is a huge flexibility here, to be looked into! Potential choices:
    #     - max specificity when half max of p within full width at half max of q (Extremely liberal)
    #     - max specificity when all of p within full width at 99.9% max of q (Extremely strict)
    if len(overlapPotVals) == 0:
        spc = 0
    else:
        overlapPosSignIdxs = np.where(overlapPotVals-overlapPotVals.max()/2 > 0)[0]
        overlapHalfMaxLeftGCN, overlapHalfMaxRightGCN = overlapGCNs[overlapPosSignIdxs[0]], overlapGCNs[overlapPosSignIdxs[-1]]
        overlapFWHM = overlapHalfMaxRightGCN - overlapHalfMaxLeftGCN
        qPosSignIdxs = np.where(q-q.max()/2 > 0)[0]
        qHalfMaxLeftGCN, qHalfMaxRightGCN = GCN_XRANGE[qPosSignIdxs[0]], GCN_XRANGE[qPosSignIdxs[-1]]
        qFWHM = qHalfMaxRightGCN - qHalfMaxLeftGCN
        if (overlapHalfMaxLeftGCN > qHalfMaxLeftGCN) and (overlapHalfMaxRightGCN < qHalfMaxRightGCN):
            spc = -0.5 / qFWHM * overlapFWHM + 1
        else:
            spc = 0.5 / qFWHM * overlapFWHM
    
    # Compute sensitivity, max when whole cluster comprises atoms with reference reaction peak GCN value, min when there is no overlapping GCN range
    sen = simps(y=p*q, x=GCN_XRANGE, axis=-1, even='avg')  # Could potentially use Romberg's method for more accurate result
    # Compute sensitivity, max when overlapping area covers peak of reference reaction distribution, min when there is no overlapping part
    # sen = 0 if spc == 0 else overlapPotVals.max() / q.max()  # Previous approach
    
    return sel, spc, sen



def calcDistDiv(p, q):
    """
    Quantify similarity between distributions p and q by computing:
        Euclidean/L2 distance between them (0 means exactly the same)
        Hellinger/Jeffreys distance between them (0 means exactly the same)
        KL divergence of distribution p from distribution q (how different is p from q, 0 means exactly similar)
        
    Inputs:
        p : NumPy array of Floats representing probability distributions
        q : NumPy array of Floats representing probability distributions
    Outputs:
        eucDist : Float representing the computed Euclidean distance
        hellDist : Float representing the computed Hellinger distance
        klDiv : Float representing the computed KL divergence
    """
    eucDist = np.sqrt(np.sum((p - q)**2))
    hellDist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    
    # Add 1% of smallest non-zero value in array to avoid division by zero error
    q.sort()
    minPosEntry = q[q>0][0]
    q += minPosEntry
    klDiv = np.sum(rel_entr(p, q))  # (in natural unit of information/nats)
    return eucDist, hellDist, klDiv


def clusterEval(X, ILSclust, verbose=False):
    """
    Evaluate the clustering results based on:
        Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index, 
        selectivity, specificity, and sensitivity towards specific reactions,
        (Removed tentatively) Euclidean distance, Hellinger distance, and Kullback-Leibler divergence from GCN profiles of specific reactions.
    
    Inputs:
        X : DataFrame with columns as features and rows as instances
        ILSclust : DataFrame (sorted by index) containing
                    cluster: Float representing ILS cluster labels
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                    gcn : Float representing GCN values of each sample atom
        verbose : Boolean determining whether print statements are activated
    Outputs:
        SH : Float indicating Silhouette Coefficient
        CH : Float indicating Calinski-Harabasz Index
        DB : Float indicating Davies-Bouldin Index
        selDictList : List of Dictionaries of Floats indicating selectivity (of each cluster toward specific reactions)
        spcDictList : List of Dictionaries of Floats indicating specificity (of each cluster toward specific reactions)
        senDictList : List of Dictionaries of Floats indicating sensitivity (of each cluster toward specific reactions)
        """
    if verbose: print("  Evaluating cluster analysis results...")
    labels = np.array(ILSclust['cluster'])
    if len(np.unique(labels)) == 1: 
        print("    Only one label, can't calculate SH, CH, and DB scores!")
        SH, CH, DB = np.nan, np.nan, np.nan
    else: 
        SH = silhouette_score(X=X, labels=labels, metric='euclidean')
        CH = calinski_harabasz_score(X=X, labels=labels)
        DB = davies_bouldin_score(X=X, labels=labels)
    if verbose: print(f"    Silhouette Coefficient: {SH:.3f}\n    Calinski-Harabasz Index: {CH:.3f}\n    Davies-Bouldin Index: {DB:.3f}")
    
    if verbose: print("  Evaluating cluster applicability/usefulness towards reference reactions based on their GCN distributions...")
    selDictList, spcDictList, senDictList = [], [], []
    for label in np.unique(labels):
        if verbose: print(f"    Computing metrics for Cluster {label} towards:")
        selDict = {reactionName: 0.0 for reactionName in refReactionsDict.keys()}
        spcDict, senDict = deepcopy(selDict), deepcopy(selDict)
        clusterGCNs = np.array(ILSclust[ILSclust['cluster'] == label]['gcn']) 
        p = FFTKDE(bw='silverman', kernel='gaussian').fit(clusterGCNs, weights=None).evaluate(GCN_XRANGE)  # TODO: Hyperparameter tuning
        
        for (i, reactionName) in enumerate(refReactionsDict.keys()):
            if verbose: print(f"      {reactionName}:")
            q = refReactionsDict[reactionName]['normPotVals']
            sel, spc, sen = calcSelSpcSen(p, q)
            selDict[reactionName], spcDict[reactionName], senDict[reactionName] = sel, spc, sen
            # eucDist, hellDist, klDiv = calcDistDiv(p, q)
            if verbose: print(f"        Selectivity: {sel:.3f}, Specificity: {spc:.3f}, Sensitivity: {sen:.3f}")
        selDictList.append(selDict)
        spcDictList.append(spcDict)
        senDictList.append(senDict)
    
    return SH, CH, DB, selDictList, spcDictList, senDictList



def visCoord3D(caseID, ILSclustAll, ILSclustSurf, showImg=False, colourFeats=['cn', 'ILSclustAll0', 'ILSclustSurf0'], verbose=False, clustMethod='Combo'):
    """
    Visualise 3D coordinates of tpdfhe nanoparticles, coloured by specified features.
    
    Inputs:
        caseID : Integer indicating index number of test case
        ILSclustAll : List of DataFrames (sorted by index) containing
                        cluster: Float representing ILS cluster labels (for all atoms)
                        gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                        gcn : Float representing GCN values of each sample atom
        ILSclustSurf : List of DataFrames (sorted by index) containing
                        cluster: Float representing ILS cluster labels (for surface atoms only)
                        gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                        gcn : Float representing GCN values of each sample atom
        showImg : Boolean indicating whether image should be shown (utility tentatively disabled)
        colourFeats : List of Strings representing features to colour the nanoparticles by
        verbose : Boolean determining whether print statements are activated
    Outputs:
        None
    """
    X = dfAlls[caseID].loc[:, :'molRMS'].copy()
    X['cn'] = X['cn'].astype('category')
    for (i, ILSclust) in enumerate(ILSclustAll):
        X[f"ClustAll{i}"] = ILSclust['cluster']
        X[f"ClustAll{i}"] = X[f"ClustAll{i}"].astype(int).astype('category')
        for colName in ILSclust.columns[3:]:
            X[f"ClustAll{colName}{i}"] = ILSclust[colName]
    for (i, ILSclust) in enumerate(ILSclustSurf):
        X[f"ClustSurf{i}"] = ILSclust['cluster']
        X[f"ClustSurf{i}"].fillna(0, inplace=True)
        X[f"ClustSurf{i}"] = X[f"ClustSurf{i}"].astype(int).astype('category')
        for colName in ILSclust.columns[3:]:
            X[f"ClustSurf{colName}{i}"] = ILSclust[colName]

    # Export DataFrame as xyz file
    visXYZfName = f"figures/ILSclust/{clustMethod}_{npName}_{testCases[caseID]}_vis80.xyz"
    with open(visXYZfName, "w") as f:
        f.write(f"{str(len(X))}\n")
        f.write(f"{testCases[caseID]}\n")
        for row in X.itertuples(name=None):
            f.write(" ".join([str(entry) for entry in row][1:]) + "\n")  # Skip index
    if verbose: print(f"Saved clustering results in {testCases[caseID]}_vis.xyz for test case {caseID}!")



# ## Clustering combination: implemente based on EAC(evidence accumulating)


def update_similar_mat(similar_mat, cluster_ind):
    '''
    input:
    similar_mat: original similarity matrix
    cluster_ind: cluster labels
    '''
    for ind, no_i in enumerate(cluster_ind): 
        if no_i == cluster_ind[-1]:
            return similar_mat
        for no_j in cluster_ind[ind+1:]:
            similar_mat[no_i, no_j] += 1
            similar_mat[no_j, no_i] += 1



def ensemble_AL(number, X, result_dict, clusterNo):
    similar_mat = np.zeros((X.shape[0], X.shape[0]))
    for single_result in result_dict:
        k_num = list(set(single_result)) 
        for k_i in k_num:
            index_same_ki = np.where(single_result == k_i)[0] 
            update_similar_mat(similar_mat, index_same_ki) 

    # 1 - similar_mat / number 
    similar_mat += np.diag([number] * X.shape[0])
    distance_mat = 1 - similar_mat / number
    distance = dist.squareform(distance_mat)
    
    # linkage SL
    Z = linkage(distance, 'average')
    #group_distance = Z[:, 2] 
    #lifetime = group_distance[1:] - group_distance[:-1] 
    #max_lifetime_index = np.argmax(lifetime)
    #threshold = Z[max_lifetime_index, 2] 

    cluster_result = fcluster(Z, t=clusterNo, criterion='maxclust')

    return cluster_result


# ## Cluster Analysis Pipeline

def runClustering(X, caseID, combID, comboResDict, allOrSurf='Surface', verbose=False, clustMethod='Combo'):
    """
    Run the cluster analysis pipeline for a dataset with specific caseID and combID.
    
    Inputs:
        X : DataFrame with each column being a feature and each row being an atom instance
        caseID : Integer indicating index number of test case
        combID : Integer indicating index number of feature combination
        comboResDict : Dictionary containing the clustering results evaluation metric for the given combination
        allOrSurf : String indicating the portion of atoms of interest {"All", "Surface"}
        verbose : Boolean determining whether print statements are activated
    Outputs:
        ILSclust : List of DataFrames (sorted by index) containing
                    cluster: Float representing ILS cluster labels
                    gamma : Float representing gamma scores of each sample, proportional to the likelihood of the point being a cluster centre
                    gcn : Float representing GCN values of each sample atom
                    AND evaluation metrics regarding cluster GCN distribution toward reference reactions
        orderedL : DataFrame from ILS plot
        comboResDict : Dictionary containing the clustering results evaluation metric for the given combination
    """
    if verbose: print(f"  Identifying densest point for {allOrSurf} atoms...")
    density, neighDensPoints, scores = calcRhoDeltaGamma(X, verbose=verbose)
    scoresIdxs = scores.argsort()[::-1]
    initL = X.index[scoresIdxs[0]]
    if verbose: print(f"    Densest point ID for {allOrSurf} atoms: {initL}")

    # Initial ILS runs
    attList = ['order', 'surf', 'order']
    if allOrSurf == 'Surface':
        attList.remove('surf')
        attList.insert(2, 'order')
    newL, orderedL = initILS(X, caseID, combID, initL, attList[:3], allOrSurf, figSize=(15, 5), verbose=verbose)
    
    # Identify the clusters by rerunning ILS with each cluster centre labelled
    gammaDF = pd.DataFrame(data=scores, index=X.index, columns=['gamma'])
    orderedL = orderedL.join(gammaDF.drop(index=initL))
    ILSclust, ILSorder = findClusts(X, caseID, combID, orderedL, allOrSurf, winSize=WINDOW_SIZE, sigConst=SIG_CONST, minClustSize=MIN_CLUST_SIZE, verbose=verbose)  # Hyperparameters!
    clusterNo = len(np.unique(np.array(ILSclust)))
    
    ILSclust.sort_index(inplace=True)    
    
    # Kmeans clustering
    KMcluster, KMcentroids = kmeans(X,clusterNo, 100)
    ILSclusterNP = np.array(ILSclust)
    KMclusterNP = np.array(KMcluster['cluster']) + 1
    ILSclust = pd.DataFrame(ILSclust).join([gammaDF, dfAlls[caseID].loc[X.index]['gcn']])
    ILSclust.rename(columns={ILSclust.columns[0]: "cluster" }, inplace = True)
    ILSclust.index.rename('ID',inplace = True)
    subClustCheck(X=X, caseID=caseID, combID=combID, orderedL=orderedL, ILSclust=ILSclust, allOrSurf=allOrSurf, verbose=verbose, clustMethod=clustMethod)  # Only makes sense for ILS
    
    # Clustering combination
    if clustMethod == 'Combo':
        result_dict = []
        number = 2
        result_dict.append(ILSclusterNP)
        result_dict.append(KMclusterNP)
        Combocluster = ensemble_AL(number, X, result_dict, clusterNo=clusterNo)
        Combocluster = pd.DataFrame(Combocluster)
        Combocluster.index = list(ILSclust.index)                    
        # ILSclust.sort_index(inplace=True)
        Combocluster = Combocluster.join([gammaDF, dfAlls[caseID].loc[X.index]['gcn']])
        Combocluster.rename(columns={Combocluster.columns[0]: "cluster" }, inplace = True)
        Combocluster.index.rename('ID',inplace = True)
    elif clustMethod == 'KMeans': 
        Combocluster = pd.Series(KMclusterNP, index=ILSclust.index)
        Combocluster = pd.DataFrame(Combocluster).join([gammaDF, dfAlls[caseID].loc[X.index]['gcn']])
        Combocluster.rename(columns={Combocluster.columns[0]: "cluster" }, inplace = True)
        Combocluster.index.rename('ID',inplace = True)
    else: Combocluster = ILSclust

    # Visualing GCN distributions for each cluster
    plotGCN(caseID=caseID, combID=combID, ILSclust=Combocluster, allOrSurf=allOrSurf, verbose=verbose, clustMethod=clustMethod)

    # Evaluate the clusters
    SHs, CHs, DBs, SELs, SPCs, SENs = clusterEval(X=X, ILSclust=Combocluster, verbose=verbose)
    for (i, metricDicts) in enumerate([SELs, SPCs, SENs]): 
        for reactionName in refReactionsDict.keys():
            Combocluster[f"{reactionName}{evalMetrics[i]}"] = Combocluster['cluster'].apply(lambda clustLabel: metricDicts[int(clustLabel)-1][reactionName])
    comboResDict[i][f"SH{allOrSurf}"], comboResDict[i][f"CH{allOrSurf}"], comboResDict[i][f"DB{allOrSurf}"] = SHs, CHs, DBs
    comboResDict[i][f"numClust{allOrSurf}"] = len(Combocluster['cluster'].unique())
    
    return Combocluster, orderedL, comboResDict



def gameCase(caseID, clustMethod=clustMethod, verbose=False):
    """
    Bundled function for experiments regarding clustering of atoms using different feature sets via game theoretical approach.
    ***Not exactly game theoretical at the moment, simply running through all combinations***
    
    Inputs:
        caseID : Integer indicating index number of test case
        verbose : Boolean determining whether print statements are activated
    Outputs:
        None
    """
    print(f"Case: {testCases[caseID]}")
    comboResDict = {i: {'testCase': '', 'bulkSurf': 0, 
                        'numAllClust': 0, 'SHAll': 0, 'CHAll': 0, 'DBAll': 0, 
                        'numSurfaceClust': 0, 'SHSurface': 0, 'CHSurface': 0, 'DBSurface': 0} 
                    for i in range(len(combList))}
    ILSclustAllList, ILSclustSurfList, bulkSurfList = [], [], []
    X_template, y_template = dfScaledNoLVHCs[caseID].iloc[:, :-1].copy(), dfScaledNoLVHCs[caseID].iloc[:, -1].copy()
    for (i, comb) in enumerate(combList):
        if i != 0: continue  # Debugging
        X, y = X_template.copy(), y_template.copy()
        X = X.round(decimals=4)
        comboResDict[i]['testCase'] = testCases[caseID]
        if len(X.columns) == 0:
            print(f"No feature left in Combination {i}! Skipping to next Combination...")
            continue
        print(f"\nRunning pipeline on Combination {i}\n  Features: {list(X.columns)}")
        
        ILSclustAll, orderedLall, comboResDict = runClustering(X, caseID, combID=i, comboResDict=comboResDict, allOrSurf='All', verbose=verbose, clustMethod=clustMethod)
        ILSclustAllList.append(ILSclustAll)
        Xsurf = X[y == 1]
        ILSclustSurf, orderedLsurf, comboResDict = runClustering(Xsurf, caseID, combID=i, comboResDict=comboResDict, allOrSurf='Surface', verbose=verbose, clustMethod=clustMethod)
        ILSclustSurfList.append(ILSclustSurf)
        
        # Check if bulk can be separated from surface automatically
        surfNo = len(Xsurf)
        bulkfNo = len(X) - len(Xsurf)
        if (bulkfNo >= surfNo):
            surfStart = int(len(Xsurf) - len(Xsurf)/SURF_ATOM_BUFFER_PERC)  # Buffer range of 10% of number of surface atoms
        else: 
            surfStart = int(len(Xsurf) + len(Xsurf)/SURF_ATOM_BUFFER_PERC)
        surfEnd = len(X) - surfStart
        bulkSurfRange = orderedLall.iloc[min(surfStart, surfEnd):max(surfStart, surfEnd), :]
        maxOrder = bulkSurfRange.iloc[bulkSurfRange['minR'].argmax(), :]['order']  # Tallest peak = surface-bulk separation in this range
        surfAtomsIdx = orderedLall.iloc[int(maxOrder):, :].index
        surfAtoms = dfScaledNoLVHCs[caseID].loc[list(surfAtomsIdx), :]
        # comboResDict[i]['bulkSurf'] = 1 if len(surfAtoms['surf'].unique()) == 1 and surfAtoms['surf'].unique()[0] == 1 else 0
        comboResDict[i]['bulkSurf'] = 1 if len(surfAtoms['surf'].unique()) == 1 else 0

    # Save clustering results in .xyz file and dictionary form
    visCoord3D(caseID=caseID, ILSclustAll=ILSclustAllList, ILSclustSurf=ILSclustSurfList, showImg=False, colourFeats=['ILSclustAll0', 'ILSclustSurf0'], clustMethod=clustMethod)

    with open(f"pickleFiles/{npName}/comboResDict{clustMethod}80_{npName}_{caseID}.pickle", 'wb') as f: pickle.dump(comboResDict, f)
    return



# Global variables that need to be run before gameCase()
# Equations that form the reference GCN-activity plots (Unnormalised)
refReactionsDict = {'AuORR':  # Au NP ORR -- oxygen reduction reaction
                    {'grad': [0.128, -0.16], 'interc': [0.201, 1.686], 'normPotVals': []}, 
                    'PtORR':  # Pt NP ORR
                    {'grad': [0.192, -0.169], 'interc': [-0.715, 2.27], 'normPotVals': []}, 
                    'PtCOOR':  # Pt COOR -- carbon monoxide oxidation reaction
                    {'grad': [0.3701, -0.1886], 'interc': [-2.464, 0.56], 'normPotVals': []}, 
                    'CuRWGSR':  # Cu RWGSR -- reverse water-gas shift reaction
                    {'grad': [0.30833, -0.1875], 'interc': [-0.88, 2.3225], 'normPotVals': []}, 
                    'CuCO2RR':  # Cu CO2RR -- carbon dioxide reduction reaction
                    {'grad': [0.163, -0.067, -0.223], 'interc': [-1.133, -0.416, 0.853], 'normPotVals': []}, 
                    'PtRCORRR':  # Pt (RCOR)RR -- aliphatic ketone reduction reaction
                    {'grad': [0.063, 0.29, -0.27, 0.055, -0.055], 'interc': [-0.35, -1.60, 1.75, -0.42, 0.42], 'normPotVals': []}} 
evalMetrics = ['Sel', 'Spc', 'Sen']

# Obtain the corresponding potential values for different reference reactions
GCN_INTERVAL = 0.001  # Modify to get different intervals of GCNs
GCN_LOW_BOUND, GCN_HIGH_BOUND = 1.0, 14.0
# Added GCN_INTERVAL because ndarray from np.arange() is open ended on right hand, NOTE: be careful to avoid evaluation point out-of-bound error for FFTKDE()
GCN_XRANGE = np.arange(start=GCN_LOW_BOUND, stop=GCN_HIGH_BOUND+GCN_INTERVAL+3, step=GCN_INTERVAL)
for reactionName in refReactionsDict.keys():
    gradList, intercList = refReactionsDict[reactionName]['grad'], refReactionsDict[reactionName]['interc']
    refReactionsDict[reactionName]['normPotVals'] = potFromGCN(gcns=GCN_XRANGE, gradList=gradList, intercList=intercList)
    
# Hyperparameters (most likely need to be tuned)
NON_POS_PEAK_FUNC_MEAN_THRESH = -0.02  # mean threshold in findPeaks()
NON_POS_PEAK_FUNC_STD_THRESH = 0.02  # standard deviation threshold in findPeaks()
SIG_CONST_SCALER = 20  # Scaler for SIG_CONST in findPeaks()
PEAK_FUNC = 'S2'  # Peak function for peak finding algorithm in locateDensestIdx()
DIFF_THRESH = 0.01  # Threshold that specifies if 2 peaks are different enough in the case of ordered nanoparticles in locateDensestIdx()
HIST_BIN_CHOICE = 3  # Choice of method to calculate number of bins or binwidth for GCN histograms in plotGCN(), currently 'doane'
WINDOW_SIZE = 20  # Window size for peak finding algorithm in runClustering()
SIG_CONST = 1.2  # Significance constant for peak finding algorithm in runClustering()
MIN_CLUST_SIZE = 10  # Minimum cluster size for clustering in runClustering()
SURF_ATOM_BUFFER_PERC = 10  # Constant for calculating the buffer range of number of surface atoms in gameCase()



if __name__ == '__main__':
    print(f"Clustering for {npName} from {startID} to {endID}, clustMethod == {clustMethod}")
    caseIDs = list(range(startID, endID))
    remainingCaseIDs = []
    for caseID in caseIDs:
        comboResDictPickleFname = f"pickleFiles/{npName}/comboResDict{clustMethod}80_{npName}_{caseID}.pickle"
        if not os.path.exists(comboResDictPickleFname): remainingCaseIDs.append(caseID)
    #remainingCaseIDs = caseIDs
    #for caseID in remainingCaseIDs: gameCase(caseID, verbose=False)
    with Pool() as p: p.map(gameCase, remainingCaseIDs)
